// SPDX-License-Identifier: BUSL-1.1

//! `restore_tenant`: validates a backup envelope, maps every backed-up
//! database to its destination, merges the sections of each database into
//! one `TenantDataSnapshot`, and re-issues every section as durable,
//! replicated writes into its destination database.

use std::sync::Arc;

use nodedb_types::backup_envelope::{
    DEFAULT_MAX_TOTAL_BYTES, EnvelopeError, parse_encrypted as parse_envelope_encrypted,
};

use crate::Error;
use crate::control::server::shared::ddl::neutral::collection::dispatch_register_from_stored;
use crate::control::state::SharedState;

use super::super::databases::{decode_databases, resolve_databases};
use super::super::sections::{apply_metadata_sections, merge_sections};
use super::rebind;
use super::stats::RestoreStats;

/// Restore a tenant from a fully-buffered backup envelope.
pub async fn restore_tenant(
    state: &Arc<SharedState>,
    tenant_id: u64,
    envelope_bytes: &[u8],
    dry_run: bool,
    force: bool,
) -> Result<RestoreStats, Error> {
    let env = match &state.backup_kek {
        Some(kek) => parse_envelope_encrypted(envelope_bytes, DEFAULT_MAX_TOTAL_BYTES, kek)?,
        None => {
            return Err(Error::Internal {
                detail: "restore: envelope is encrypted but no backup KEK is configured; \
                         set [backup_encryption] in the server config"
                    .into(),
            });
        }
    };
    if env.meta.tenant_id != tenant_id {
        return Err(EnvelopeError::TenantMismatch {
            expected: tenant_id,
            actual: env.meta.tenant_id,
        }
        .into());
    }

    // Every group the restore reads or writes has a reachable majority, or
    // the restore fails here, before it proposes anything.
    if !dry_run {
        super::super::quorum::require_quorum(state)?;
    }

    let newest = if !dry_run && env.meta.snapshot_watermark != 0 {
        super::super::guard::newest_committed_write(state, tenant_id).await?
    } else {
        None
    };
    if let Some(mark) = newest {
        let current_high_water = mark.hlc;
        if env.meta.snapshot_watermark < current_high_water {
            if force {
                tracing::warn!(
                    tenant_id,
                    envelope_watermark = env.meta.snapshot_watermark,
                    current_high_water,
                    newest_write_site = mark.site.as_str(),
                    newest_write_collection = mark.collection.as_deref().unwrap_or(""),
                    "restore staleness protection explicitly overridden via FORCE: \
                     envelope watermark is older than the destination cluster's last \
                     observed write-HLC for this tenant — newer writes will be overwritten"
                );
            } else {
                return Err(Error::Internal {
                    detail: format!(
                        "restore refused: envelope watermark {} is older than the \
                         destination cluster's last observed write-HLC {} for tenant \
                         {} (newest write: {} on collection '{}') — newer writes would \
                         be silently overwritten",
                        env.meta.snapshot_watermark,
                        current_high_water,
                        tenant_id,
                        mark.site,
                        mark.collection.as_deref().unwrap_or("<none>"),
                    ),
                });
            }
        }
    }

    let mut stats = RestoreStats {
        tenant_id,
        dry_run,
        sections: env.sections.len() as u16,
        source_vshard_count: env.meta.source_vshard_count,
        ..Default::default()
    };

    // Map every backed-up database to its destination, creating each one the
    // destination lacks. Every other section names its database by source id.
    let database_blobs = decode_databases(&env)?;
    let databases = resolve_databases(state, tenant_id, &database_blobs, dry_run)?;
    stats.databases = database_blobs.len();
    stats.databases_created = databases.created();

    if !dry_run {
        let restored_collections = apply_metadata_sections(state, tenant_id, &env, &databases)?;
        // Every restored collection's declaration reaches this node's Data
        // Plane before any of its rows do. The catalog row alone leaves
        // `doc_configs` empty for the collection, and the re-issue below
        // would then ingest a timeseries collection's rows into an inferred
        // shape: the declared time key becomes an integer field and the row
        // is stamped with the restore-time clock. This is the same
        // registration a committed DDL and the boot rehydration dispatch,
        // and it replaces any registration already present, so a cluster
        // applier's own register hook and a later boot seed are both
        // idempotent with it. A registration failure fails the restore.
        for coll in &restored_collections {
            // A classified error keeps its class. Only a machinery failure
            // gains the restore context.
            dispatch_register_from_stored(state, coll)
                .await
                .map_err(|e| {
                    if crate::error_classify::is_unclassified_failure(&e) {
                        Error::Internal {
                            detail: format!(
                                "restore: Data Plane registration of collection '{}' failed: {e}",
                                coll.name
                            ),
                        }
                    } else {
                        e
                    }
                })?;
        }
    }

    let merged = merge_sections(&env.sections)?;
    for source in merged.keys() {
        if !database_blobs
            .iter()
            .any(|blob| blob.database_id == *source)
        {
            return Err(Error::Internal {
                detail: format!(
                    "invalid backup format: a data section names database {source}, which the \
                     backup's database section does not list"
                ),
            });
        }
    }
    for (source, snap) in &merged {
        stats.count_sections(snap);
        if dry_run {
            stats.columnar_engines += snap.columnar_engines.len();
        }
        // A dry run has no target for a database this cluster lacks: no
        // tombstone of this cluster names it.
        if let Some(target) = databases.get(*source) {
            rebind::warn_on_tombstoned_restores(
                state,
                tenant_id,
                target,
                snap,
                env.meta.snapshot_watermark,
            );
        }
    }

    if dry_run {
        return Ok(stats);
    }

    // Each database re-issues its rows into its destination database.
    for (source, snap) in merged {
        let target = databases.target(source)?;
        super::database::reissue_database(state, tenant_id, target, snap, &mut stats).await?;
    }
    Ok(stats)
}
