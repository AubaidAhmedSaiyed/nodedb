// SPDX-License-Identifier: Apache-2.0

//! `EnvelopeWriter` — builds a plaintext backup envelope byte-by-byte.

use super::types::{
    DEFAULT_MAX_SECTION_BYTES, DEFAULT_MAX_TOTAL_BYTES, HEADER_LEN, MAGIC, SECTION_OVERHEAD,
    TRAILER_LEN, VERSION,
};
use super::types::{EnvelopeError, EnvelopeMeta, Section};

/// Build an envelope by pushing sections one at a time, then `finalize()`.
pub struct EnvelopeWriter {
    pub(super) meta: EnvelopeMeta,
    pub(super) sections: Vec<Section>,
    pub(super) max_total: u64,
    pub(super) max_section: u64,
    pub(super) framed_size: u64,
}

impl EnvelopeWriter {
    pub fn new(meta: EnvelopeMeta) -> Self {
        Self::with_caps(meta, DEFAULT_MAX_TOTAL_BYTES, DEFAULT_MAX_SECTION_BYTES)
    }

    pub fn with_caps(meta: EnvelopeMeta, max_total: u64, max_section: u64) -> Self {
        Self {
            meta,
            sections: Vec::new(),
            max_total,
            max_section,
            framed_size: HEADER_LEN as u64 + TRAILER_LEN as u64,
        }
    }

    pub fn push_section(
        &mut self,
        origin_node_id: u64,
        body: Vec<u8>,
    ) -> Result<(), EnvelopeError> {
        if body.len() as u64 > self.max_section {
            return Err(EnvelopeError::OverSizeSection {
                cap: self.max_section,
            });
        }
        let added = SECTION_OVERHEAD as u64 + body.len() as u64;
        if self.framed_size + added > self.max_total {
            return Err(EnvelopeError::OverSizeTotal {
                cap: self.max_total,
            });
        }
        if self.sections.len() >= u16::MAX as usize {
            return Err(EnvelopeError::TooManySections(u16::MAX));
        }
        self.framed_size += added;
        self.sections.push(Section {
            origin_node_id,
            body,
        });
        Ok(())
    }

    /// Finalize without encryption. Produces a plaintext envelope of [`VERSION`].
    pub fn finalize(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.framed_size as usize);
        write_header(&mut out, &self.meta, self.sections.len() as u16, VERSION);
        for section in &self.sections {
            write_section(&mut out, section);
        }
        // Trailer crc covers header bytes + every section's framed bytes.
        let trailer_crc = crc32c::crc32c(&out);
        out.extend_from_slice(&trailer_crc.to_le_bytes());
        out
    }
}

pub(super) fn write_header(
    out: &mut Vec<u8>,
    meta: &EnvelopeMeta,
    section_count: u16,
    version: u8,
) {
    let start = out.len();
    out.extend_from_slice(MAGIC); // [0..4]
    out.push(version); // [4]
    out.extend_from_slice(&[0u8; 3]); // [5..8]  _reserved
    out.extend_from_slice(&meta.tenant_id.to_le_bytes()); // [8..16] tenant_id (u64)
    out.extend_from_slice(&meta.source_vshard_count.to_le_bytes()); // [16..18]
    out.extend_from_slice(&[0u8; 6]); // [18..24] _reserved
    out.extend_from_slice(&meta.hash_seed.to_le_bytes()); // [24..32]
    out.extend_from_slice(&meta.snapshot_watermark.to_le_bytes()); // [32..40]
    out.extend_from_slice(&section_count.to_le_bytes()); // [40..42]
    out.extend_from_slice(&[0u8; 6]); // [42..48] _reserved
    let header_crc = crc32c::crc32c(&out[start..]);
    out.extend_from_slice(&header_crc.to_le_bytes()); // [48..52]
}

pub(super) fn write_section(out: &mut Vec<u8>, section: &Section) {
    out.extend_from_slice(&section.origin_node_id.to_le_bytes());
    out.extend_from_slice(&(section.body.len() as u32).to_le_bytes());
    out.extend_from_slice(&section.body);
    let body_crc = crc32c::crc32c(&section.body);
    out.extend_from_slice(&body_crc.to_le_bytes());
}
