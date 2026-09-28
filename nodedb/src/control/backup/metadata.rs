// SPDX-License-Identifier: BUSL-1.1

//! The tenant's databases and the metadata sections of its backup.
//!
//! A tenant's collections can live in any database. The backup covers every
//! database the tenant has a collection in. The metadata sections record, per
//! database:
//!
//! - the database itself: its descriptor, its quota and the tenant's quota in
//!   it (`SECTION_ORIGIN_DATABASES`);
//! - the catalog row of each of the tenant's collections in it
//!   (`SECTION_ORIGIN_CATALOG_ROWS`);
//! - the PK-to-surrogate binds of those collections
//!   (`SECTION_ORIGIN_SURROGATE_PK`);
//! - the WAL tombstones of the tenant's purged collections in it
//!   (`SECTION_ORIGIN_SOURCE_TOMBSTONES`).
//!
//! Every entry names its database by the source id. Restore maps each source
//! id to a destination database by name.

use std::collections::{BTreeMap, BTreeSet};

use nodedb_types::backup_envelope::{
    DatabaseBlob, EnvelopeWriter, SECTION_ORIGIN_CATALOG_ROWS, SECTION_ORIGIN_DATABASES,
    SECTION_ORIGIN_SOURCE_TOMBSTONES, SECTION_ORIGIN_SURROGATE_PK, SourceTombstoneEntry,
    StoredCollectionBlob, SurrogateBindBlob,
};

use crate::Error;
use crate::control::security::catalog::{DatabaseDescriptor, StoredCollection, SystemCatalog};
use crate::control::state::SharedState;
use crate::types::{DatabaseId, TenantId};

/// One database the tenant has collections in.
pub struct TenantDatabase {
    pub descriptor: DatabaseDescriptor,
    /// Every collection of the tenant in this database, soft-deleted ones
    /// included: UNDROP works after a restore of a backup taken during the
    /// retention window.
    pub collections: Vec<StoredCollection>,
}

impl TenantDatabase {
    pub fn id(&self) -> DatabaseId {
        self.descriptor.id
    }
}

/// Every database `tenant_id` has a collection in, in database-id order.
///
/// A collection whose database has no catalog entry fails the backup: the
/// restore could not recreate that database, and its rows would be lost.
pub fn tenant_databases(state: &SharedState, tenant_id: u64) -> Result<Vec<TenantDatabase>, Error> {
    let catalog = state.credentials.catalog();
    let mut by_database: BTreeMap<u64, Vec<StoredCollection>> = BTreeMap::new();
    for coll in catalog.load_all_collections_across_databases()? {
        if coll.tenant_id == tenant_id {
            by_database
                .entry(coll.database_id.as_u64())
                .or_default()
                .push(coll);
        }
    }
    let mut databases = Vec::with_capacity(by_database.len());
    for (raw_id, collections) in by_database {
        let descriptor = catalog
            .get_database(DatabaseId::new(raw_id))?
            .ok_or_else(|| Error::Internal {
                detail: format!(
                    "backup: tenant {tenant_id} has collections in database {raw_id}, \
                         but the catalog has no entry for that database. Restore the \
                         database entry, then retry the backup"
                ),
            })?;
        databases.push(TenantDatabase {
            descriptor,
            collections,
        });
    }
    Ok(databases)
}

/// Push the metadata sections of `databases`. A catalog read or encode error
/// fails the backup: an envelope without these sections restores rows into
/// no database, rows a point lookup cannot find, or a purged collection.
pub fn push_metadata_sections(
    state: &SharedState,
    tenant_id: u64,
    databases: &[TenantDatabase],
    writer: &mut EnvelopeWriter,
) -> Result<(), Error> {
    let catalog = state.credentials.catalog();
    let tenant = TenantId::new(tenant_id);

    let mut blobs = Vec::with_capacity(databases.len());
    for database in databases {
        blobs.push(DatabaseBlob {
            database_id: database.id().as_u64(),
            name: database.descriptor.name.clone(),
            descriptor: encode_section_part("database descriptor", &database.descriptor)?,
            database_quota: catalog.get_database_quota(database.id())?,
            tenant_quota: catalog.get_tenant_quota(database.id(), tenant)?,
        });
    }
    push_nonempty(writer, SECTION_ORIGIN_DATABASES, "databases", &blobs)?;

    let mut rows = Vec::new();
    for database in databases {
        for coll in &database.collections {
            rows.push(StoredCollectionBlob {
                database_id: database.id().as_u64(),
                name: coll.name.clone(),
                bytes: encode_section_part("catalog row", coll)?,
            });
        }
    }
    push_nonempty(writer, SECTION_ORIGIN_CATALOG_ROWS, "catalog rows", &rows)?;

    // PK→surrogate identity map. This is DATA-derived per-node state that the
    // per-node engine sections do NOT carry (the Data-Plane snapshot handler
    // has no catalog access). Without it a restored node has documents but
    // cannot resolve PK point-lookups (`WHERE id=<pk>`).
    let binds = surrogate_binds(catalog, tenant, databases)?;
    push_nonempty(writer, SECTION_ORIGIN_SURROGATE_PK, "surrogate pk", &binds)?;

    let backed_up: BTreeSet<u64> = databases.iter().map(|d| d.id().as_u64()).collect();
    let mut tombs = Vec::new();
    for (database_id, tid, name, purge_lsn) in catalog.load_wal_tombstones()?.iter() {
        if tid == tenant_id && backed_up.contains(&database_id) {
            tombs.push(SourceTombstoneEntry {
                database_id,
                collection: name.to_string(),
                purge_lsn,
            });
        }
    }
    push_nonempty(
        writer,
        SECTION_ORIGIN_SOURCE_TOMBSTONES,
        "source tombstones",
        &tombs,
    )
}

/// Every PK→surrogate bind of the tenant's collections in `databases`.
fn surrogate_binds(
    catalog: &SystemCatalog,
    tenant: TenantId,
    databases: &[TenantDatabase],
) -> Result<Vec<SurrogateBindBlob>, Error> {
    let mut binds = Vec::new();
    for database in databases {
        for coll in &database.collections {
            let rows = catalog.scan_surrogates_for_collection(
                nodedb_types::CollectionKey::from_bare(database.id(), &coll.name),
                tenant,
            )?;
            for (pk, surrogate) in rows {
                binds.push(SurrogateBindBlob {
                    database_id: database.id().as_u64(),
                    tenant_id: tenant.as_u64(),
                    collection: coll.name.clone(),
                    pk,
                    surrogate: surrogate.as_u32(),
                });
            }
        }
    }
    Ok(binds)
}

/// Encode one part of a section.
pub(super) fn encode_section_part<T: zerompk::ToMessagePack>(
    what: &str,
    value: &T,
) -> Result<Vec<u8>, Error> {
    zerompk::to_msgpack_vec(value).map_err(|e| Error::Serialization {
        format: "msgpack".into(),
        detail: format!("backup envelope ({what}): encode: {e}"),
    })
}

/// Encode `entries` and push them as the section `origin`, unless empty.
fn push_nonempty<T: zerompk::ToMessagePack>(
    writer: &mut EnvelopeWriter,
    origin: u64,
    what: &str,
    entries: &[T],
) -> Result<(), Error> {
    if entries.is_empty() {
        return Ok(());
    }
    let body = encode_section_part(what, &entries)?;
    writer
        .push_section(origin, body)
        .map_err(|e| Error::Internal {
            detail: format!("backup envelope ({what}): {e}"),
        })
}
