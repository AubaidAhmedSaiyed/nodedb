// SPDX-License-Identifier: BUSL-1.1

//! Vector-primary row fixtures shared by the direct-write and replay tests.

use std::collections::HashMap;

use nodedb_physical::physical_plan::VectorDirectWriteIntent;
use nodedb_types::{RlsWriteCheck, Surrogate, Value};

use crate::bridge::envelope::Status;
use crate::data::executor::core_loop::CoreLoop;
use crate::data::executor::core_loop::tests::make_default_task;
use crate::data::executor::handlers::vector_upsert::VectorDirectUpsertParams;
use crate::types::DatabaseId;

pub const TID: u64 = 1;
pub const COLLECTION: &str = "vp_rows";
pub const FIELD: &str = "emb";

/// Store one row under `surrogate`.
pub fn seed(core: &mut CoreLoop, surrogate: Surrogate) {
    let mut fields: HashMap<String, Value> = HashMap::new();
    fields.insert("owner".into(), Value::String("a".into()));
    let payload = zerompk::to_msgpack_vec(&fields).expect("encode payload");
    let task = make_default_task();
    let resp = core.execute_vector_direct_upsert(VectorDirectUpsertParams {
        task: &task,
        tid: TID,
        collection: COLLECTION,
        field: FIELD,
        surrogate,
        vector: &[1.0, 0.0, 0.0],
        payload: &payload,
        quantization: nodedb_types::VectorQuantization::None,
        storage_dtype: nodedb_types::VectorStorageDtype::F32,
        payload_indexes: &[],
        intent: VectorDirectWriteIntent::Insert,
        on_conflict_updates: &[],
        rls_write_check: &RlsWriteCheck::decided_earlier_in_request(),
        returning: None,
        rls_filters: &[],
    });
    assert_eq!(resp.status, Status::Ok, "seed: {:?}", resp.error_code);
}

/// Whether the index holds a node for `surrogate`.
pub fn has_node(core: &CoreLoop, surrogate: Surrogate) -> bool {
    let index_key =
        CoreLoop::vector_index_key(DatabaseId::DEFAULT.as_u64(), TID, COLLECTION, FIELD);
    core.vector_direct_node(&index_key, surrogate).is_some()
}

/// Whether the sidecar store holds `surrogate`'s row.
pub fn has_sidecar(core: &CoreLoop, surrogate: Surrogate) -> bool {
    core.vector_sidecar_row(DatabaseId::DEFAULT.as_u64(), TID, COLLECTION, surrogate)
        .expect("read sidecar")
        .is_some()
}
