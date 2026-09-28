// SPDX-License-Identifier: Apache-2.0

//! Canonical collection identity for placement and surrogate identity.
//!
//! A collection's vShard and its surrogate bindings are keyed by the pair
//! `(database_id, bare_name)`. The bare name is the name the catalog keys the
//! collection by. The database-qualified form `"{database_id}/{name}"` names
//! the same collection, but it folds the database into the string a second
//! time. Hashing it gives a different vShard than hashing the bare name.
//!
//! [`CollectionKey`] is the only input the vShard hash and the surrogate
//! allocator accept. It has no `From<&str>`. A caller builds it from a bare
//! catalog name with [`CollectionKey::from_bare`], or from a qualified name
//! with [`CollectionKey::from_qualified`], which strips the qualifier.

use super::{DatabaseId, QualifiedCollection};

/// Error returned when a qualified collection name does not carry the
/// qualifier of the database it is resolved in.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CollectionKeyError {
    /// The name lacks the `"{database_id}/"` prefix that
    /// [`QualifiedCollection::new`] writes for a non-default database.
    #[error(
        "collection name '{name}' is not qualified for database {database_id}; \
         expected the prefix '{database_id}/'"
    )]
    NotQualified {
        /// Raw id of the database the name was resolved in.
        database_id: u64,
        /// The rejected name.
        name: String,
    },
}

/// The canonical `(database_id, bare_name)` identity of a collection.
///
/// Borrowed: it never allocates. Build it with [`Self::from_bare`] from a
/// catalog name, or with [`Self::from_qualified`] /
/// [`Self::from_qualified_str`] from a database-qualified name.
///
/// No `From<&str>` impl exists, so a raw string never converts implicitly:
///
/// ```compile_fail
/// use nodedb_types::CollectionKey;
/// let key: CollectionKey<'_> = "1024/users".into();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollectionKey<'a> {
    database_id: DatabaseId,
    name: &'a str,
}

impl<'a> CollectionKey<'a> {
    /// Build a key from the bare name the catalog keys the collection by.
    ///
    /// `name` must be the catalog name, never a `"{database_id}/{name}"`
    /// string. A qualified name goes through [`Self::from_qualified`].
    pub fn from_bare(database_id: DatabaseId, name: &'a str) -> Self {
        Self { database_id, name }
    }

    /// Build a key from a [`QualifiedCollection`], stripping its qualifier.
    pub fn from_qualified(
        database_id: DatabaseId,
        qualified: &'a QualifiedCollection,
    ) -> Result<Self, CollectionKeyError> {
        Self::from_qualified_str(database_id, qualified.as_str())
    }

    /// Build a key from a database-qualified name carried as a string on a
    /// plan, a WAL record, or the wire.
    ///
    /// The default database stores names unqualified, so the name is taken
    /// as-is. The empty name marks a plan with no routing collection and is
    /// the same in both forms, so it is also taken as-is. Any other name in
    /// any other database requires the exact `"{database_id}/"` prefix
    /// [`QualifiedCollection::new`] writes. Exactly one prefix is stripped,
    /// so a bare name that itself contains `/` survives intact.
    pub fn from_qualified_str(
        database_id: DatabaseId,
        qualified: &'a str,
    ) -> Result<Self, CollectionKeyError> {
        if database_id == DatabaseId::DEFAULT || qualified.is_empty() {
            return Ok(Self::from_bare(database_id, qualified));
        }
        let not_qualified = || CollectionKeyError::NotQualified {
            database_id: database_id.as_u64(),
            name: qualified.to_owned(),
        };
        let (head, bare) = qualified.split_once('/').ok_or_else(not_qualified)?;
        if head.is_empty() || !head.bytes().all(|b| b.is_ascii_digit()) {
            return Err(not_qualified());
        }
        match head.parse::<u64>() {
            Ok(raw) if raw == database_id.as_u64() && !head.starts_with('0') => {
                Ok(Self::from_bare(database_id, bare))
            }
            _ => Err(not_qualified()),
        }
    }

    /// The database the collection lives in.
    pub fn database_id(&self) -> DatabaseId {
        self.database_id
    }

    /// The bare catalog name.
    pub fn name(&self) -> &'a str {
        self.name
    }

    /// The database-qualified name storage engines key data by.
    pub fn qualified(&self) -> QualifiedCollection {
        QualifiedCollection::new(self.database_id, self.name)
    }

    /// The vShard this collection homes to.
    pub fn vshard(&self) -> super::VShardId {
        super::VShardId::from_collection(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DB: DatabaseId = DatabaseId::new(1024);

    #[test]
    fn qualified_input_yields_the_bare_key() {
        let qualified = QualifiedCollection::new(DB, "users");
        let from_qualified = CollectionKey::from_qualified(DB, &qualified).expect("qualified");
        assert_eq!(from_qualified.name(), "users");
        assert_eq!(from_qualified, CollectionKey::from_bare(DB, "users"));
        assert_eq!(
            from_qualified.vshard(),
            CollectionKey::from_bare(DB, "users").vshard()
        );
    }

    #[test]
    fn qualified_string_cannot_become_a_key_without_dequalifying() {
        // The only string-accepting constructors are `from_bare`, which the
        // caller names explicitly, and the qualified constructors, which strip
        // the qualifier. The qualified path therefore always lands on the bare
        // name, never on the qualified string.
        let qualified = QualifiedCollection::new(DB, "users");
        let key = CollectionKey::from_qualified_str(DB, qualified.as_str()).expect("qualified");
        assert_ne!(key.name(), qualified.as_str());
        assert_eq!(key.name(), "users");
        assert_eq!(key.qualified(), qualified);
    }

    #[test]
    fn default_database_names_are_already_bare() {
        let key = CollectionKey::from_qualified_str(DatabaseId::DEFAULT, "users").expect("default");
        assert_eq!(key, CollectionKey::from_bare(DatabaseId::DEFAULT, "users"));
    }

    #[test]
    fn empty_name_is_the_no_collection_sentinel_in_every_database() {
        let key = CollectionKey::from_qualified_str(DB, "").expect("empty");
        assert_eq!(key, CollectionKey::from_bare(DB, ""));
    }

    #[test]
    fn exactly_one_qualifier_is_stripped() {
        let qualified = QualifiedCollection::new(DB, "1024/nested");
        let key = CollectionKey::from_qualified(DB, &qualified).expect("qualified");
        assert_eq!(key.name(), "1024/nested");
    }

    #[test]
    fn unqualified_name_in_a_named_database_is_rejected() {
        let err = CollectionKey::from_qualified_str(DB, "users").expect_err("unqualified");
        assert_eq!(
            err,
            CollectionKeyError::NotQualified {
                database_id: 1024,
                name: "users".to_owned(),
            }
        );
    }

    #[test]
    fn foreign_database_qualifier_is_rejected() {
        assert!(CollectionKey::from_qualified_str(DB, "7/users").is_err());
        assert!(CollectionKey::from_qualified_str(DB, "+1024/users").is_err());
        assert!(CollectionKey::from_qualified_str(DB, "01024/users").is_err());
        assert!(CollectionKey::from_qualified_str(DB, "/users").is_err());
    }

    #[test]
    fn bare_and_qualified_hashes_differ_for_a_named_database() {
        // The two strings name one collection. Only the key keeps them on one
        // vShard, so hashing the qualified string directly is a routing error.
        let mut differs = false;
        for i in 0..64 {
            let name = format!("coll_{i}");
            let qualified = QualifiedCollection::new(DB, &name);
            let key = CollectionKey::from_qualified(DB, &qualified).expect("qualified");
            let raw_qualified = CollectionKey::from_bare(DB, qualified.as_str());
            assert_eq!(key.vshard(), CollectionKey::from_bare(DB, &name).vshard());
            differs |= key.vshard() != raw_qualified.vshard();
        }
        assert!(differs);
    }
}
