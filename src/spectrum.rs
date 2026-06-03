use crate::graph::Graph;
use crate::sheaf::SheafConfig;
use crate::flow::FlowFn;
use nalgebra::DMatrix;

/// Computed spectrum at a single time point.
#[derive(Debug, Clone)]
pub struct Spectrum {
    /// Time point.
    pub t: f64,
    /// All eigenvalues, sorted ascending.
    pub eigenvalues: Vec<f64>,
    /// Smallest nonzero eigenvalue (spectral gap λ₁).
    pub lambda1: f64,
    /// Second nonzero eigenvalue (λ₂).
    pub lambda2: f64,
    /// Largest eigenvalue.
    pub max_eigenvalue: f64,
    /// Trace of the Laplacian.
    pub trace: f64,
}

impl Spectrum {
    /// Number of eigenvalues.
    pub fn n_eigenvalues(&self) -> usize {
        self.eigenvalues.len()
    }
}

/// Compute the Hodge Laplacian spectrum for the sheaf at time t.
///
/// The Laplacian is built as:
///   L[i,i] += r², L[j,j] += r², L[i,j] -= r², L[j,i] -= r²
/// for each edge (i,j) with restriction r = cfg.eval_restriction(flow(e, t)) * weight.
pub fn compute_spectrum(
    g: &Graph,
    cfg: &SheafConfig,
    flow: FlowFn,
    t: f64,
) -> Spectrum {
    let n = g.n_vertices;
    let mut laplacian = DMatrix::<f64>::zeros(n, n);

    for (e_idx, edge) in g.edges.iter().enumerate() {
        let i = edge.src;
        let j = edge.dst;
        let e_energy = flow(e_idx, t);
        let r = cfg.eval_restriction(e_energy) * edge.weight;

        laplacian[(i, i)] += r * r;
        laplacian[(j, j)] += r * r;
        laplacian[(i, j)] -= r * r;
        laplacian[(j, i)] -= r * r;
    }

    // Use Jacobi rotation for symmetric eigenvalue decomposition
    let eigenvalues = jacobi_eigenvalues(&laplacian, 100 * n);
    let max_eigenvalue = eigenvalues[n - 1];
    let trace: f64 = eigenvalues.iter().sum();

    // Find λ₁ = smallest nonzero eigenvalue, λ₂ = second nonzero
    let mut lambda1 = -1.0;
    let mut lambda2 = -1.0;
    let mut found = false;
    for &ev in &eigenvalues {
        if ev > 1e-10 {
            if !found {
                lambda1 = ev;
                found = true;
            } else {
                lambda2 = ev;
                break;
            }
        }
    }
    if lambda2 < 0.0 {
        lambda2 = lambda1;
    }

    Spectrum {
        t,
        eigenvalues,
        lambda1,
        lambda2,
        max_eigenvalue,
        trace,
    }
}

/// Jacobi rotation eigenvalue solver for symmetric matrices.
///
/// Iteratively zeroes out off-diagonal elements using Givens rotations.
fn jacobi_eigenvalues(mat: &DMatrix<f64>, max_sweeps: usize) -> Vec<f64> {
    let n = mat.nrows();
    let mut a = mat.clone();

    for _sweep in 0..max_sweeps {
        // Find the largest off-diagonal element (above diagonal)
        let mut max_off = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[(i, j)].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < 1e-14 * n as f64 {
            break;
        }

        // Compute rotation angle
        let app = a[(p, p)];
        let aqq = a[(q, q)];
        let apq = a[(p, q)];
        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation: A' = G^T A G
        for i in 0..n {
            if i == p || i == q {
                continue;
            }
            let mip = a[(i, p)];
            let miq = a[(i, q)];
            a[(i, p)] = c * mip + s * miq;
            a[(p, i)] = a[(i, p)];
            a[(i, q)] = -s * mip + c * miq;
            a[(q, i)] = a[(i, q)];
        }

        let new_pp = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        let new_qq = s * s * app - 2.0 * s * c * apq + c * c * aqq;

        a[(p, p)] = new_pp;
        a[(q, q)] = new_qq;
        a[(p, q)] = 0.0;
        a[(q, p)] = 0.0;
    }

    // Extract diagonal and sort
    let mut evals: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    evals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::flow_constant;
    use crate::graph::Graph;

    const TOL: f64 = 0.05;

    #[test]
    fn test_static_cycle4_known_values() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::static_sheaf(1.0);
        let sp = compute_spectrum(&g, &cfg, flow_constant, 0.0);

        assert_eq!(sp.n_eigenvalues(), 4);
        assert!((sp.eigenvalues[0] - 0.0).abs() < TOL, "kernel eigenvalue ~ 0");
        assert!((sp.lambda1 - 2.0).abs() < TOL, "λ₁ ≈ 2.0 for C4, got {}", sp.lambda1);
        assert!((sp.max_eigenvalue - 4.0).abs() < TOL, "λ_max ≈ 4.0, got {}", sp.max_eigenvalue);
    }

    #[test]
    fn test_static_trace_formula() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::static_sheaf(2.0);
        let sp = compute_spectrum(&g, &cfg, flow_constant, 0.0);
        // 4 edges, R=2: trace = 4 * 2 * 4 = 32
        assert!((sp.trace - 32.0).abs() < 0.5, "trace ≈ 32, got {}", sp.trace);
    }

    #[test]
    fn test_static_gap_scales_r0_squared() {
        let g = Graph::cycle(4);
        let sp1 = compute_spectrum(&g, &SheafConfig::static_sheaf(1.0), flow_constant, 0.0);
        let sp2 = compute_spectrum(&g, &SheafConfig::static_sheaf(2.0), flow_constant, 0.0);
        let ratio = sp2.lambda1 / sp1.lambda1;
        assert!((ratio - 4.0).abs() < 0.1, "gap scales as R₀²: ratio={}", ratio);
    }

    #[test]
    fn test_static_gap_constant_over_time() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::static_sheaf(1.0);
        let gaps: Vec<f64> = (0..5)
            .map(|i| {
                let sp = compute_spectrum(&g, &cfg, crate::flow::flow_sinusoidal, i as f64 * 2.0);
                sp.lambda1
            })
            .collect();
        for i in 1..5 {
            assert!((gaps[i] - gaps[0]).abs() < 0.001,
                "static gap constant over time: i={}", i);
        }
    }

    #[test]
    fn test_single_edge() {
        let edges = vec![crate::graph::Edge::new(0, 1, 1.0)];
        let g = Graph::new(2, edges);
        let cfg = SheafConfig::static_sheaf(1.0);
        let sp = compute_spectrum(&g, &cfg, flow_constant, 0.0);
        assert_eq!(sp.n_eigenvalues(), 2);
        assert!((sp.eigenvalues[0] - 0.0).abs() < TOL, "kernel");
        assert!(sp.lambda1 > 0.0, "positive gap");
    }

    #[test]
    fn test_large_cycle_20() {
        let g = Graph::cycle(20);
        let cfg = SheafConfig::linear(1.0, 0.3);
        let sp = compute_spectrum(&g, &cfg, crate::flow::flow_sinusoidal, 2.0);
        assert_eq!(sp.n_eigenvalues(), 20);
        assert!(sp.lambda1 > 0.0, "positive gap");
    }

    #[test]
    fn test_path_gap() {
        let g = Graph::path(4);
        let cfg = SheafConfig::static_sheaf(1.0);
        let sp = compute_spectrum(&g, &cfg, flow_constant, 0.0);
        assert!(sp.lambda1 > 0.0);
        assert!(sp.lambda1 < 2.0, "path gap < cycle gap for same |V|");
    }

    #[test]
    fn test_complete_gap() {
        let g = Graph::complete(5);
        let cfg = SheafConfig::static_sheaf(1.0);
        let sp = compute_spectrum(&g, &cfg, flow_constant, 0.0);
        assert!(sp.lambda1 > 3.0, "K5 gap > 3.0 (good expansion)");
    }

    #[test]
    fn test_complete_vs_cycle_gap() {
        let c4 = Graph::cycle(4);
        let k4 = Graph::complete(4);
        let cfg = SheafConfig::static_sheaf(1.0);
        let cs = compute_spectrum(&c4, &cfg, flow_constant, 0.0);
        let ks = compute_spectrum(&k4, &cfg, flow_constant, 0.0);
        assert!(ks.lambda1 > cs.lambda1, "K4 gap > C4 gap");
    }

    #[test]
    fn test_linear_gap_differs_from_static() {
        let g = Graph::cycle(4);
        let ss = compute_spectrum(&g, &SheafConfig::static_sheaf(1.0),
                                  crate::flow::flow_sinusoidal, 5.0);
        let ls = compute_spectrum(&g, &SheafConfig::linear(1.0, 0.5),
                                  crate::flow::flow_sinusoidal, 5.0);
        assert!((ls.lambda1 - ss.lambda1).abs() > 0.01,
                "linear evolving gap differs from static: diff={}",
                (ls.lambda1 - ss.lambda1).abs());
    }

    #[test]
    fn test_linear_alpha_zero_equals_static() {
        let g = Graph::cycle(4);
        let ss = compute_spectrum(&g, &SheafConfig::static_sheaf(1.0),
                                  crate::flow::flow_sinusoidal, 3.0);
        let ls = compute_spectrum(&g, &SheafConfig::linear(1.0, 0.0),
                                  crate::flow::flow_sinusoidal, 3.0);
        assert!((ss.lambda1 - ls.lambda1).abs() < 0.001, "α=0 → same as static");
    }

    #[test]
    fn test_nonlinear_sigmoid_vs_static() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::nonlinear(2.0, crate::sheaf::NonlinearFn::Sigmoid, 2.0);
        let sp = compute_spectrum(&g, &cfg, crate::flow::flow_sinusoidal, 2.0);
        assert!(sp.lambda1 > 0.0, "sigmoid gap > 0");
    }

    #[test]
    fn test_nonlinear_tanh_gap() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::nonlinear(2.0, crate::sheaf::NonlinearFn::Tanh, 1.0);
        let sp = compute_spectrum(&g, &cfg, crate::flow::flow_sinusoidal, 0.0);
        assert!(sp.lambda1 > 0.0);
    }

    #[test]
    fn test_spectrum_ordering() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::static_sheaf(1.0);
        let sp = compute_spectrum(&g, &cfg, flow_constant, 0.0);
        for i in 1..sp.n_eigenvalues() {
            assert!(sp.eigenvalues[i - 1] <= sp.eigenvalues[i] + 1e-10,
                    "eigenvalues sorted ascending");
        }
    }

    #[test]
    fn test_symmetric_laplacian() {
        let g = Graph::cycle(10);
        let cfg = SheafConfig::linear(1.0, 0.5);
        let n = g.n_vertices;
        let mut lap = DMatrix::<f64>::zeros(n, n);

        for (e_idx, edge) in g.edges.iter().enumerate() {
            let e_energy = crate::flow::flow_sinusoidal(e_idx, 1.0);
            let r = cfg.eval_restriction(e_energy) * edge.weight;
            lap[(edge.src, edge.src)] += r * r;
            lap[(edge.dst, edge.dst)] += r * r;
            lap[(edge.src, edge.dst)] -= r * r;
            lap[(edge.dst, edge.src)] -= r * r;
        }

        // Upper and lower parts should match
        for i in 0..n {
            for j in 0..n {
                assert!((lap[(i, j)] - lap[(j, i)]).abs() < 1e-12,
                        "Laplacian symmetric at ({},{})", i, j);
            }
        }
    }
}
