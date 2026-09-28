// SPDX-License-Identifier: Apache-2.0

//! k-means++ seeding and Lloyd iterations for the IVF coarse quantizer.

use crate::distance::{DistanceMetric, distance};

/// Up to `k` centroids for `data`, seeded by k-means++ from a fixed seed so
/// the same training set always yields the same centroids.
pub(crate) fn kmeans_centroids(
    data: &[&[f32]],
    dim: usize,
    k: usize,
    max_iter: usize,
) -> Vec<Vec<f32>> {
    let n = data.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }

    let mut centroids: Vec<Vec<f32>> = vec![data[0].to_vec()];
    let mut min_dists = vec![f32::MAX; n];

    // Initialize min_dists against the first centroid.
    for (i, point) in data.iter().enumerate() {
        let d = distance(point, &centroids[0], DistanceMetric::L2);
        if d < min_dists[i] {
            min_dists[i] = d;
        }
    }

    let mut rng = crate::hnsw::Xorshift64::new(0xC0FF_EEDE_ADBE_EF42);
    for _ in 1..k {
        let total: f64 = min_dists.iter().map(|&d| d as f64).sum();
        let next_idx = if total < f64::EPSILON {
            0
        } else {
            let target = rng.next_f64() * total;
            let mut acc = 0.0f64;
            let mut chosen = n - 1;
            for (i, &d) in min_dists.iter().enumerate() {
                acc += d as f64;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        let last = data[next_idx];
        centroids.push(last.to_vec());
        for (i, point) in data.iter().enumerate() {
            let d = distance(point, last, DistanceMetric::L2);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
    }

    let mut assignments = vec![0usize; n];
    for _ in 0..max_iter {
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_d = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = distance(point, centroid, DistanceMetric::L2);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, point) in data.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += point[d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
        }
    }
    centroids
}
