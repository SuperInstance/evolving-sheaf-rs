/// Flow energy functions that drive the evolution of restriction maps.
pub type FlowFn = fn(edge_idx: usize, t: f64) -> f64;

/// Constant flow: always returns 1.0.
pub fn flow_constant(_edge: usize, _t: f64) -> f64 {
    1.0
}

/// Sinusoidal flow: 1.0 + 0.5·sin(0.5·t + edge·0.3).
pub fn flow_sinusoidal(edge: usize, t: f64) -> f64 {
    1.0 + 0.5 * (0.5 * t + edge as f64 * 0.3).sin()
}

/// Pulse flow: alternates between high (3.0) and low (0.5).
pub fn flow_pulse(edge: usize, t: f64) -> f64 {
    let period = 4.0;
    let phase = t.rem_euclid(period);
    let pw = 0.3 + 0.05 * (edge % 5) as f64;
    if phase < pw { 3.0 } else { 0.5 }
}

/// A deterministic "random" walk based on a hash-like seed.
pub fn flow_random_walk(edge: usize, t: f64) -> f64 {
    let seed = (t * 100.0) as i64 + edge as i64 * 137;
    // Simple deterministic hash
    let s = (seed.wrapping_mul(2654435761)) as u64;
    let s = ((s >> 16) ^ s).wrapping_mul(0x45d9f3b);
    let s = ((s >> 16) ^ s).wrapping_mul(0x45d9f3b);
    let s = (s >> 16) ^ s;
    0.5 + (s & 0x7FFF_FFFF) as f64 / 0x7FFF_FFFF as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_flow() {
        let v = flow_constant(0, 0.0);
        assert!((v - 1.0).abs() < 1e-12);
        let v = flow_constant(5, 100.0);
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sinusoidal_range() {
        for edge in 0..10 {
            for t in [0.0, 1.0, 10.0, 100.0] {
                let v = flow_sinusoidal(edge, t);
                assert!(v >= 0.5 && v <= 1.5, "sinusoidal in [0.5, 1.5]");
            }
        }
    }

    #[test]
    fn test_pulse_oscillation() {
        let v_low = flow_pulse(0, 0.0);
        let v_high = flow_pulse(0, 0.15);
        let v_low2 = flow_pulse(0, 2.0);
        assert!(v_high >= v_low, "pulse has high and low phases");
        assert!((v_low - 0.5).abs() < 1e-6 || (v_low - 3.0).abs() < 1e-6);
        assert!((v_low2 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_random_walk_range() {
        for edge in 0..5 {
            for t in [0.0, 1.0, 10.0] {
                let v = flow_random_walk(edge, t);
                assert!(v >= 0.5 && v < 1.5, "random walk in [0.5, 1.5)");
            }
        }
    }

    #[test]
    fn test_random_walk_deterministic() {
        let v1 = flow_random_walk(3, 7.0);
        let v2 = flow_random_walk(3, 7.0);
        assert!((v1 - v2).abs() < 1e-12, "deterministic walk");
    }
}
