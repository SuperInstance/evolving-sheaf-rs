use crate::graph::Graph;
use crate::sheaf::SheafConfig;
use crate::flow::FlowFn;
use crate::spectrum::compute_spectrum;

/// A 2D stability map of spectral gap over (coupling_alpha, time) grid.
///
/// Rows correspond to alpha values, columns to time points.
#[derive(Debug, Clone)]
pub struct StabilityMap {
    /// Alpha values (rows).
    pub alphas: Vec<f64>,
    /// Time values (columns).
    pub times: Vec<f64>,
    /// Gap values stored row-major: gap_values[i * n_times + j].
    pub gap_values: Vec<f64>,
}

impl StabilityMap {
    pub fn n_alphas(&self) -> usize {
        self.alphas.len()
    }

    pub fn n_times(&self) -> usize {
        self.times.len()
    }

    /// Get the gap value at (alpha_idx, time_idx).
    pub fn get(&self, alpha_idx: usize, time_idx: usize) -> f64 {
        self.gap_values[alpha_idx * self.n_times() + time_idx]
    }

    /// Compute the total variation for each alpha row.
    pub fn row_variation(&self) -> Vec<f64> {
        let n_t = self.n_times();
        self.alphas.iter().enumerate().map(|(i, _)| {
            let base = self.get(i, 0);
            (1..n_t).map(|j| (self.get(i, j) - base).abs()).sum()
        }).collect()
    }
}

/// Compute a stability map: spectral gap over (alpha, time) grid.
///
/// Parameters:
/// - `g`: graph topology
/// - `base_cfg`: base sheaf config (alpha will be overridden per row)
/// - `flow`: flow energy function
/// - `alphas`: array of alpha values to scan
/// - `times`: array of time points to scan
///
/// Returns a `StabilityMap` with gap_values stored row-major.
pub fn stability_map(
    g: &Graph,
    base_cfg: &SheafConfig,
    flow: FlowFn,
    alphas: &[f64],
    times: &[f64],
) -> StabilityMap {
    let n_alphas = alphas.len();
    let n_times = times.len();
    let mut gap_values = vec![0.0; n_alphas * n_times];

    for (i, &alpha) in alphas.iter().enumerate() {
        let mut cfg = *base_cfg;
        cfg.alpha = alpha;
        for (j, &t) in times.iter().enumerate() {
            let sp = compute_spectrum(g, &cfg, flow, t);
            gap_values[i * n_times + j] = sp.lambda1;
        }
    }

    StabilityMap {
        alphas: alphas.to_vec(),
        times: times.to_vec(),
        gap_values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::flow::*;
    use crate::sheaf::SheafConfig;

    #[test]
    fn test_stability_map_basic() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.0);

        let alphas = vec![0.0, 0.5, 1.0];
        let times = vec![0.0, 5.0, 10.0];

        let sm = stability_map(&g, &cfg, flow_sinusoidal, &alphas, &times);

        assert_eq!(sm.n_alphas(), 3);
        assert_eq!(sm.n_times(), 3);
        assert_eq!(sm.gap_values.len(), 9);

        // α=0 should be constant across time
        assert!((sm.get(0, 0) - sm.get(0, 1)).abs() < 0.01);
        assert!((sm.get(0, 0) - sm.get(0, 2)).abs() < 0.01);

        // Larger α should differ from α=0
        let any_diff = (0..3).any(|j| (sm.get(2, j) - sm.get(0, 0)).abs() > 0.01);
        assert!(any_diff, "larger α changes gap");
    }

    #[test]
    fn test_stability_map_increasing_alpha() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.0);

        let alphas = vec![0.0, 2.0];
        let times = vec![0.0, 3.0, 6.0, 9.0];

        let sm = stability_map(&g, &cfg, flow_sinusoidal, &alphas, &times);

        let var0 = (1..4).map(|j| (sm.get(0, j) - sm.get(0, 0)).abs()).sum::<f64>();
        let var1 = (1..4).map(|j| (sm.get(1, j) - sm.get(1, 0)).abs()).sum::<f64>();

        assert!(var0 < 0.01, "α=0: no variation");
        assert!(var1 > var0, "α=2: more variation than α=0");
    }

    #[test]
    fn test_stability_map_grid_dims() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::static_sheaf(1.0);

        let alphas: Vec<f64> = (0..5).map(|i| i as f64 * 0.5).collect();
        let times: Vec<f64> = (0..10).map(|i| i as f64).collect();

        let sm = stability_map(&g, &cfg, flow_sinusoidal, &alphas, &times);
        assert_eq!(sm.n_alphas(), 5);
        assert_eq!(sm.n_times(), 10);
        assert_eq!(sm.gap_values.len(), 50);
    }

    #[test]
    fn test_stability_row_variation() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.0);
        let alphas = vec![0.0, 1.0];
        let times = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let sm = stability_map(&g, &cfg, flow_sinusoidal, &alphas, &times);

        let vars = sm.row_variation();
        assert_eq!(vars.len(), 2);
        assert!(vars[1] > vars[0], "larger alpha → more row variation");
    }

    #[test]
    fn test_stability_map_with_constant_flow() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.5);
        let alphas = vec![0.0, 1.0];
        let times = vec![0.0, 5.0, 10.0];
        let sm = stability_map(&g, &cfg, flow_constant, &alphas, &times);

        // With constant flow, gap should be same across time at each alpha
        assert!((sm.get(0, 0) - sm.get(0, 1)).abs() < 0.001);
        assert!((sm.get(0, 0) - sm.get(0, 2)).abs() < 0.001);
        assert!((sm.get(1, 0) - sm.get(1, 1)).abs() < 0.001);
        assert!((sm.get(1, 0) - sm.get(1, 2)).abs() < 0.001);
    }

    #[test]
    fn test_stability_map_all_positive() {
        let g = Graph::cycle(5);
        let cfg = SheafConfig::linear(1.0, 0.0);
        let alphas = vec![0.0, 0.5, 1.0, 2.0];
        let times = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let sm = stability_map(&g, &cfg, flow_sinusoidal, &alphas, &times);

        // All gap values should be positive
        for &val in &sm.gap_values {
            assert!(val > 0.0, "all gaps positive: got {}", val);
        }
    }
}
