// SPDX-License-Identifier: Apache-2.0

//! Sub-vector distance and k-means for PQ codebook training.

/// L2 squared distance for sub-vectors (used in k-means and encoding).
#[inline]
pub(crate) fn l2_sub(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Simple k-means clustering for PQ codebook training.
///
/// Uses proper k-means++ initialization (weighted d² sampling) with a
/// deterministic seed so training is reproducible across runs.
pub(crate) fn kmeans(data: &[&[f32]], dim: usize, k: usize, max_iter: usize) -> Vec<Vec<f32>> {
    let n = data.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let k = k.min(n); // Can't have more centroids than data points.

    // K-means++ initialization with deterministic xorshift.
    let mut rng = crate::hnsw::Xorshift64::new(0xC0FF_EEDE_ADBE_EF42);

    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(data[0].to_vec());

    let mut min_dists = vec![f32::MAX; n];
    // Update against the first centroid.
    for (i, point) in data.iter().enumerate() {
        let d = l2_sub(point, &centroids[0]);
        if d < min_dists[i] {
            min_dists[i] = d;
        }
    }

    for _ in 1..k {
        let total: f64 = min_dists.iter().map(|&d| d as f64).sum();
        let next_idx = if total < f64::EPSILON {
            // All points coincide with existing centroids.
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
        // Incrementally update min_dists against the new centroid.
        for (i, point) in data.iter().enumerate() {
            let d = l2_sub(point, last);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
    }

    // K-means iterations.
    let mut assignments = vec![0usize; n];
    for _ in 0..max_iter {
        // Assignment step.
        let mut changed = false;
        for (i, point) in data.iter().enumerate() {
            let mut best = 0;
            let mut best_d = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = l2_sub(point, centroid);
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

        // Update step: recompute centroids as means.
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
