// SPDX-License-Identifier: BUSL-1.1

//! Vector collection construction and post-write settling.
//!
//! Every vector index, whatever its type, is one `VectorCollection` built
//! from the index configuration `CREATE VECTOR INDEX` set. After a write the
//! collection settles: a full growing segment seals and its HNSW build request
//! goes to `build_tx`, and an IVF-PQ collection that holds its training
//! threshold trains and moves its buffered vectors into the IVF-PQ index.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use tracing::{error, info, warn};

use crate::data::executor::core_loop::CoreLoop;
use crate::data::executor::handlers::vector_direct_row::VectorIndexKey;
use crate::engine::vector::collection::VectorCollection;
use crate::engine::vector::hnsw::HnswParams;
use crate::engine::vector::index_config::{IndexConfig, IndexType};

/// The configuration a new collection under `key` is built with: the index
/// configuration when one was set, else the default index type over the
/// HNSW parameters set for `key`.
pub(in crate::data::executor) fn vector_index_config_for(
    index_configs: &HashMap<VectorIndexKey, IndexConfig>,
    vector_params: &HashMap<VectorIndexKey, HnswParams>,
    key: &VectorIndexKey,
) -> IndexConfig {
    match index_configs.get(key) {
        Some(config) => config.clone(),
        None => IndexConfig {
            hnsw: vector_params.get(key).cloned().unwrap_or_default(),
            ..IndexConfig::default()
        },
    }
}

/// An IVF-PQ index splits each vector into `pq_m` equal subvectors, so its
/// dimension must be a multiple of `pq_m`. Any other index type passes.
pub(in crate::data::executor) fn check_ivf_dim(
    config: &IndexConfig,
    dim: usize,
) -> crate::Result<()> {
    if config.index_type == IndexType::IvfPq
        && (config.pq_m == 0 || !dim.is_multiple_of(config.pq_m))
    {
        return Err(crate::Error::DataException {
            detail: format!(
                "an ivf_pq index needs a vector dimension divisible by pq_m {}; got dimension {dim}",
                config.pq_m
            ),
        });
    }
    Ok(())
}

impl CoreLoop {
    /// Create the collection under `key` at `dim` when it does not exist,
    /// from the index configuration registered under `config_key`. A
    /// schemaless field index registers under the bare collection key and
    /// stores under the field-qualified one. Fails as [`check_ivf_dim`] does.
    pub(in crate::data::executor) fn ensure_vector_collection(
        &mut self,
        key: &VectorIndexKey,
        config_key: &VectorIndexKey,
        dim: usize,
    ) -> crate::Result<&mut VectorCollection> {
        match self.vector_collections.entry(key.clone()) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let config =
                    vector_index_config_for(&self.index_configs, &self.vector_params, config_key);
                check_ivf_dim(&config, dim)?;
                Ok(entry.insert(VectorCollection::with_index_config(dim, config)))
            }
        }
    }

    /// Settle the collection under `key` after a write: seal a full growing
    /// segment and send its HNSW build, or train an IVF-PQ collection that
    /// holds its training threshold.
    pub(in crate::data::executor) fn settle_vector_collection(&mut self, key: &VectorIndexKey) {
        let seal_key = CoreLoop::vector_build_key(key);
        let Some(coll) = self.vector_collections.get_mut(key) else {
            return;
        };
        if coll.needs_seal()
            && let Some(req) = coll.seal(&seal_key)
            && let Some(tx) = &self.build_tx
            && let Err(e) = tx.send(req)
        {
            warn!(core = self.core_id, error = %e, "failed to send HNSW build request");
        }
        self.train_ivf_if_ready(key);
    }

    /// Train the collection under `key` when it is an untrained IVF-PQ
    /// collection holding its training threshold.
    pub(in crate::data::executor) fn train_ivf_if_ready(&mut self, key: &VectorIndexKey) {
        if self
            .vector_collections
            .get(key)
            .is_some_and(|coll| coll.needs_ivf_training())
        {
            self.train_vector_collection_ivf(key);
        }
    }

    /// Train every IVF-PQ collection that holds its training threshold. Boot
    /// runs it once WAL replay and the store rebuild have restored the
    /// buffers, so a collection that crossed its threshold before a restart
    /// searches through IVF-PQ again without waiting for a write.
    pub fn train_ready_ivf_collections(&mut self) {
        let ready: Vec<VectorIndexKey> = self
            .vector_collections
            .iter()
            .filter(|(_, coll)| coll.needs_ivf_training())
            .map(|(key, _)| key.clone())
            .collect();
        for key in ready {
            self.train_vector_collection_ivf(&key);
        }
    }

    /// Train the IVF-PQ index of the collection under `key`.
    ///
    /// The write that crossed the threshold is already applied and durable,
    /// so a training error does not fail it. The vectors stay in the exact
    /// buffer, search stays correct, and the next write retries the training.
    fn train_vector_collection_ivf(&mut self, key: &VectorIndexKey) {
        let memory = nodedb_mem::ScopedMemory::new(
            self.governor.clone(),
            key.0,
            key.1,
            nodedb_mem::EngineId::Vector,
        );
        // no-determinism: the wall clock only stamps the training for
        // `SHOW VECTOR INDEX`; no stored state or search depends on it.
        let trained_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
        let Some(coll) = self.vector_collections.get_mut(key) else {
            return;
        };
        let threshold = coll.ivf_training_threshold();
        match coll.train_ivf(memory, trained_at_ms) {
            Ok(()) => {
                let trained_on = coll.ivf_index().map_or(0, |ivf| ivf.trained_on());
                info!(
                    core = self.core_id,
                    key = %key.2,
                    threshold,
                    trained_on,
                    "IVF-PQ index trained; buffered vectors moved into it"
                );
                self.checkpoint_coordinator.mark_dirty("vector", trained_on);
            }
            Err(e) => {
                error!(
                    core = self.core_id,
                    key = %key.2,
                    threshold,
                    error = %e,
                    "IVF-PQ training failed; vectors stay in the exact buffer and the next write retries"
                );
            }
        }
    }
}
