

/// Evolution mode for the sheaf's restriction maps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvolutionMode {
    /// R₀ constant — theorem holds; spectral gap stays fixed.
    Static,
    /// R(t) = R₀ + α·E(t); gap changes linearly with flow energy.
    Linear,
    /// R(t) = R₀ · f(E(t)), f = sigmoid/tanh/exp-decay; complex dynamics.
    Nonlinear,
}

/// Nonlinear activation function selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonlinearFn {
    Sigmoid,
    Tanh,
    ExpDecay,
}

impl NonlinearFn {
    pub fn apply(&self, x: f64, k: f64) -> f64 {
        match self {
            NonlinearFn::Sigmoid => 1.0 / (1.0 + (-k * x).exp()),
            NonlinearFn::Tanh => (k * x).tanh(),
            NonlinearFn::ExpDecay => (-k * x * x).exp(),
        }
    }
}

/// Configuration for the sheaf's restriction map model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SheafConfig {
    pub model: EvolutionMode,
    /// Base restriction map magnitude.
    pub r0: f64,
    /// Linear scaling coefficient.
    pub alpha: f64,
    /// Nonlinear function selector.
    pub nonlin: NonlinearFn,
    /// Steepness parameter for nonlinear functions.
    pub nonlin_k: f64,
}

impl Default for SheafConfig {
    fn default() -> Self {
        SheafConfig {
            model: EvolutionMode::Static,
            r0: 1.0,
            alpha: 0.0,
            nonlin: NonlinearFn::Sigmoid,
            nonlin_k: 1.0,
        }
    }
}

impl SheafConfig {
    /// Evaluate the restriction map for a given flow energy.
    pub fn eval_restriction(&self, flow_energy: f64) -> f64 {
        match self.model {
            EvolutionMode::Static => self.r0,
            EvolutionMode::Linear => self.r0 + self.alpha * flow_energy,
            EvolutionMode::Nonlinear => self.r0 * self.nonlin.apply(flow_energy, self.nonlin_k),
        }
    }

    /// Create a static sheaf config.
    pub fn static_sheaf(r0: f64) -> Self {
        SheafConfig {
            model: EvolutionMode::Static,
            r0,
            alpha: 0.0,
            nonlin: NonlinearFn::Sigmoid,
            nonlin_k: 1.0,
        }
    }

    /// Create a linear evolving sheaf config.
    pub fn linear(r0: f64, alpha: f64) -> Self {
        SheafConfig {
            model: EvolutionMode::Linear,
            r0,
            alpha,
            nonlin: NonlinearFn::Sigmoid,
            nonlin_k: 1.0,
        }
    }

    /// Create a nonlinear evolving sheaf config.
    pub fn nonlinear(r0: f64, nonlin: NonlinearFn, k: f64) -> Self {
        SheafConfig {
            model: EvolutionMode::Nonlinear,
            r0,
            alpha: 0.0,
            nonlin,
            nonlin_k: k,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_zero() {
        let v = NonlinearFn::Sigmoid.apply(0.0, 1.0);
        assert!((v - 0.5).abs() < 1e-6, "sigmoid(0) = 0.5");
    }

    #[test]
    fn test_tanh_zero() {
        let v = NonlinearFn::Tanh.apply(0.0, 1.0);
        assert!((v - 0.0).abs() < 1e-6, "tanh(0) = 0");
    }

    #[test]
    fn test_expdecay_zero() {
        let v = NonlinearFn::ExpDecay.apply(0.0, 1.0);
        assert!((v - 1.0).abs() < 1e-6, "exp(-0) = 1");
    }

    #[test]
    fn test_sigmoid_large_positive() {
        let v = NonlinearFn::Sigmoid.apply(100.0, 1.0);
        assert!((v - 1.0).abs() < 1e-6, "sigmoid(∞) → 1");
    }

    #[test]
    fn test_sigmoid_large_negative() {
        let v = NonlinearFn::Sigmoid.apply(-100.0, 1.0);
        assert!((v - 0.0).abs() < 1e-6, "sigmoid(-∞) → 0");
    }

    #[test]
    fn test_eval_static() {
        let cfg = SheafConfig::static_sheaf(3.14);
        let r = cfg.eval_restriction(5.0);
        assert!((r - 3.14).abs() < 1e-6, "static R = R₀");
    }

    #[test]
    fn test_eval_linear() {
        let cfg = SheafConfig::linear(1.0, 0.5);
        let r = cfg.eval_restriction(2.0);
        assert!((r - 2.0).abs() < 1e-6, "linear: 1.0 + 0.5*2.0 = 2.0");
    }

    #[test]
    fn test_eval_nonlinear_tanh() {
        let cfg = SheafConfig::nonlinear(2.0, NonlinearFn::Tanh, 1.0);
        let r = cfg.eval_restriction(1.0);
        let expected = 2.0 * (1.0_f64).tanh();
        assert!((r - expected).abs() < 1e-6, "nonlinear: 2.0*tanh(1.0)");
    }

    #[test]
    fn test_eval_nonlinear_sigmoid() {
        let cfg = SheafConfig::nonlinear(2.0, NonlinearFn::Sigmoid, 1.0);
        let r = cfg.eval_restriction(0.0);
        // R₀ * sigmoid(0) = 2.0 * 0.5 = 1.0
        assert!((r - 1.0).abs() < 1e-6, "nonlinear sigmoid(0) = 1.0");
    }

    #[test]
    fn test_default() {
        let cfg: SheafConfig = Default::default();
        assert_eq!(cfg.model, EvolutionMode::Static);
        assert!((cfg.r0 - 1.0).abs() < 1e-12);
        assert!((cfg.alpha - 0.0).abs() < 1e-12);
    }
}
