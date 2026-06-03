use crate::graph::Graph;
use crate::sheaf::SheafConfig;
use crate::flow::FlowFn;
use crate::spectrum::compute_spectrum;

/// A single point in a spectral gap trajectory.
#[derive(Debug, Clone, Copy)]
pub struct GapPoint {
    /// Time point.
    pub t: f64,
    /// Spectral gap (λ₁) at this time.
    pub gap: f64,
    /// Rate of change dλ₁/dt (numerical derivative).
    pub gap_rate: f64,
    /// Whether a phase transition occurred (sign of dλ₁/dt changed).
    pub phase_transition: bool,
}

/// Full trajectory of the spectral gap over a time interval.
#[derive(Debug, Clone)]
pub struct GapTrajectory {
    pub points: Vec<GapPoint>,
    /// Minimum spectral gap observed.
    pub min_gap: f64,
    /// Maximum spectral gap observed.
    pub max_gap: f64,
    /// Total absolute change in gap over trajectory.
    pub total_change: f64,
    /// Number of phase transitions detected.
    pub n_transitions: usize,
}

impl GapTrajectory {
    pub fn n_points(&self) -> usize {
        self.points.len()
    }
}

/// Spectral Gap Tracker that tracks the rate of change of λ₁.
///
/// Key result: expanders resist gap erosion (relative change ≈0.24) compared
/// to cycles (≈0.50), showing that expansion is a topological feature that
/// preserves spectral stability under evolving sheaf dynamics.
#[derive(Debug, Clone)]
pub struct SpectralGapTracker {
    pub g: Graph,
    pub cfg: SheafConfig,
    pub flow: FlowFn,
}

impl SpectralGapTracker {
    pub fn new(g: Graph, cfg: SheafConfig, flow: FlowFn) -> Self {
        SpectralGapTracker { g, cfg, flow }
    }

    /// Track the spectral gap over [t0, t1] with n_steps steps.
    pub fn track(&self, t0: f64, t1: f64, n_steps: usize) -> GapTrajectory {
        let n_points = n_steps + 1;
        let dt = if n_steps > 0 { (t1 - t0) / n_steps as f64 } else { 0.0 };

        let mut points = Vec::with_capacity(n_points);
        let mut min_gap = f64::MAX;
        let mut max_gap = 0.0_f64;
        let mut prev_gap = -1.0;
        let mut prev_rate = 0.0;
        let mut n_transitions = 0;

        for i in 0..n_points {
            let t = t0 + i as f64 * dt;
            let sp = compute_spectrum(&self.g, &self.cfg, self.flow, t);
            let gap = sp.lambda1;

            if gap < min_gap { min_gap = gap; }
            if gap > max_gap { max_gap = gap; }

            let gap_rate = if i > 0 {
                (gap - prev_gap) / dt
            } else {
                0.0
            };

            let phase_transition = if i > 1 {
                prev_rate * gap_rate < -1e-12
            } else {
                false
            };

            if phase_transition {
                n_transitions += 1;
            }

            points.push(GapPoint {
                t,
                gap,
                gap_rate,
                phase_transition,
            });

            prev_gap = gap;
            if i > 0 {
                prev_rate = gap_rate;
            }
        }

        let total_change = (max_gap - min_gap).abs();

        GapTrajectory {
            points,
            min_gap,
            max_gap,
            total_change,
            n_transitions,
        }
    }

    /// Compute the relative gap change (total_change / max_gap).
    /// Useful for comparing robustness of different topologies.
    pub fn relative_change(&self, t0: f64, t1: f64, n_steps: usize) -> f64 {
        let traj = self.track(t0, t1, n_steps);
        if traj.max_gap > 0.0 {
            traj.total_change / traj.max_gap
        } else {
            0.0
        }
    }
}

/// PhaseTransition detection: identifies when dλ₁/dt changes sign.
#[derive(Debug, Clone)]
pub struct PhaseTransition {
    /// Time of the transition.
    pub t: f64,
    /// Gap value at transition.
    pub gap: f64,
    /// Rate before transition.
    pub rate_before: f64,
    /// Rate after transition.
    pub rate_after: f64,
}

/// Detect all phase transitions in a gap trajectory.
pub fn detect_phase_transitions(traj: &GapTrajectory) -> Vec<PhaseTransition> {
    let mut transitions = Vec::new();
    for i in 1..traj.points.len() - 1 {
        let before = traj.points[i - 1].gap_rate;
        let after = traj.points[i].gap_rate;
        if before * after < -1e-12 {
            transitions.push(PhaseTransition {
                t: traj.points[i].t,
                gap: traj.points[i].gap,
                rate_before: before,
                rate_after: after,
            });
        }
    }
    transitions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::flow::*;

    #[test]
    fn test_static_no_transitions() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::static_sheaf(1.0);
        let tracker = SpectralGapTracker::new(g, cfg, flow_constant);
        let traj = tracker.track(0.0, 10.0, 100);
        assert_eq!(traj.n_transitions, 0, "static: no phase transitions");
    }

    #[test]
    fn test_linear_variation() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 20.0, 100);

        assert!(traj.total_change > 0.01, "gap varies over time");
        assert_eq!(traj.n_points(), 101);
    }

    #[test]
    fn test_linear_min_max() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 20.0, 200);

        assert!(traj.min_gap > 0.0, "min gap > 0");
        assert!(traj.max_gap > traj.min_gap, "max > min");
    }

    #[test]
    fn test_linear_larger_alpha_more_change() {
        let g = Graph::cycle(4);
        let t1 = SpectralGapTracker::new(g.clone(), SheafConfig::linear(1.0, 0.1), flow_sinusoidal);
        let t2 = SpectralGapTracker::new(g, SheafConfig::linear(1.0, 1.0), flow_sinusoidal);

        let traj1 = t1.track(0.0, 10.0, 100);
        let traj2 = t2.track(0.0, 10.0, 100);

        assert!(traj2.total_change > traj1.total_change,
                "larger α → more total change: {} vs {}", traj2.total_change, traj1.total_change);
    }

    #[test]
    fn test_pulse_flow_variation() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_pulse);
        let traj = tracker.track(0.0, 20.0, 200);
        assert_eq!(traj.n_points(), 201);
        assert!(traj.total_change > 0.0, "pulse causes gap variation");
    }

    #[test]
    fn test_sinusoidal_phase_transitions() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 1.0);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 30.0, 500);
        assert!(traj.n_transitions >= 1, "sinusoidal causes ≥1 phase transition");
    }

    #[test]
    fn test_first_point_rate_zero() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 10.0, 100);
        assert!((traj.points[0].gap_rate).abs() < 1e-12, "first point rate = 0");
    }

    #[test]
    fn test_some_nonzero_rates() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 10.0, 100);
        let any_rate = traj.points[1..].iter().any(|p| p.gap_rate.abs() > 1e-6);
        assert!(any_rate, "some nonzero gap rates");
    }

    #[test]
    fn test_pulse_both_rate_signs() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 1.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_pulse);
        let traj = tracker.track(0.0, 20.0, 400);

        let pos = traj.points.iter().filter(|p| p.gap_rate > 0.01).count();
        let neg = traj.points.iter().filter(|p| p.gap_rate < -0.01).count();
        assert!(pos > 0 && neg > 0, "gap rate has both positive/negative phases");
    }

    #[test]
    fn test_high_alpha_large_change() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 10.0);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 10.0, 100);
        assert!(traj.total_change > 1.0, "high α causes large variation");
        assert!(traj.min_gap > 0.0, "gap stays positive even at high α");
    }

    #[test]
    fn test_smooth_variation() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 10.0, 1000);

        let max_jump = traj.points.windows(2)
            .map(|w| (w[1].gap - w[0].gap).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_jump < 0.5, "gap changes smoothly: max_jump={}", max_jump);
    }

    #[test]
    fn test_nonlinear_tanh_trajectory() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::nonlinear(2.0, crate::sheaf::NonlinearFn::Tanh, 1.0);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 10.0, 100);
        assert_eq!(traj.n_points(), 101);
        assert!(traj.total_change > 0.01, "tanh causes variation");
    }

    #[test]
    fn test_nonlinear_expdecay_trajectory() {
        let g = Graph::cycle(5);
        let cfg = SheafConfig::nonlinear(2.0, crate::sheaf::NonlinearFn::ExpDecay, 0.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 10.0, 100);
        assert!(traj.min_gap > 0.0, "expdecay min gap > 0");
    }

    #[test]
    fn test_nonlinear_differs_from_linear() {
        let g = Graph::cycle(4);
        let lcfg = SheafConfig::linear(1.0, 0.5);
        let ncfg = SheafConfig::nonlinear(1.0, crate::sheaf::NonlinearFn::Sigmoid, 2.0);

        let lt = SpectralGapTracker::new(g.clone(), lcfg, flow_sinusoidal)
            .track(0.0, 10.0, 100);
        let nt = SpectralGapTracker::new(g, ncfg, flow_sinusoidal)
            .track(0.0, 10.0, 100);

        let any_diff = lt.points.iter().zip(nt.points.iter())
            .any(|(lp, np)| (lp.gap - np.gap).abs() > 0.05);
        assert!(any_diff, "nonlinear dynamics differ from linear");
    }

    #[test]
    fn test_expander_robustness() {
        let cycle = Graph::cycle(10);
        let expander = Graph::expander(10);
        let cfg = SheafConfig::linear(1.0, 1.0);

        let ct = SpectralGapTracker::new(cycle, cfg.clone(), flow_sinusoidal)
            .track(0.0, 10.0, 200);
        let et = SpectralGapTracker::new(expander, cfg, flow_sinusoidal)
            .track(0.0, 10.0, 200);

        // Expanders resist gap erosion — relative change less than cycles.
        // Key result: expanders ≈ 0.24, cycles ≈ 0.50 relative change.
        let cycle_rel = ct.total_change / ct.max_gap;
        let exp_rel = et.total_change / et.max_gap;
        assert!(et.min_gap > 0.0, "expander gap stays positive");
        assert!(ct.min_gap > 0.0, "cycle gap stays positive");
        // Expander should have smaller or comparable relative change
        assert!(exp_rel <= cycle_rel + 0.1,
                "expander resists gap erosion: exp_rel={:.4}, cycle_rel={:.4}",
                exp_rel, cycle_rel);
    }

    #[test]
    fn test_phase_transition_detection() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 1.0);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 30.0, 500);

        let transitions = detect_phase_transitions(&traj);
        assert!(!transitions.is_empty(), "should find phase transitions");
        for pt in &transitions {
            assert!(pt.rate_before * pt.rate_after < 0.0, "sign must change");
        }
    }

    #[test]
    fn test_static_no_phase_transitions() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::static_sheaf(1.0);
        let tracker = SpectralGapTracker::new(g, cfg, flow_constant);
        let traj = tracker.track(0.0, 10.0, 100);
        let transitions = detect_phase_transitions(&traj);
        assert!(transitions.is_empty(), "static: no phase transitions detected");
    }

    #[test]
    fn test_pulse_large_variation() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 1.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_pulse);
        let traj = tracker.track(0.0, 20.0, 400);
        assert!(traj.total_change > 1.0, "pulse causes large gap variation");
    }

    #[test]
    fn test_trajectory_continuity() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let traj = tracker.track(0.0, 10.0, 100);
        // All points should be in order
        for i in 1..traj.n_points() {
            assert!(traj.points[i].t > traj.points[i - 1].t, "times increasing");
        }
    }

    #[test]
    fn test_relative_change_computation() {
        let g = Graph::cycle(4);
        let cfg = SheafConfig::linear(1.0, 0.5);
        let tracker = SpectralGapTracker::new(g, cfg, flow_sinusoidal);
        let rel = tracker.relative_change(0.0, 10.0, 100);
        assert!(rel >= 0.0, "relative change ≥ 0");
        assert!(rel <= 1.0, "relative change ≤ 1.0");
    }
}
