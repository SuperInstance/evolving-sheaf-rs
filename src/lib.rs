//! # evolving-sheaf (Rust port)
//!
//! Spectral Gap Dynamics in Evolving Cellular Sheaves.
//!
//! Studies how the Hodge Laplacian's spectral gap (λ₁) changes when restriction maps
//! become flow-dependent. Static sheaves preserve the gap; dynamic ones reveal phase
//! transitions.
//!
//! ## Model
//!
//! The sheaf Hodge Laplacian is L = D†D where D is the coboundary operator whose entries
//! depend on restriction maps. Restriction maps may be static (theorem holds) or dynamic
//! (gap decreases — research!).

pub mod graph;
pub mod sheaf;
pub mod flow;
pub mod spectrum;
pub mod trajectory;
pub mod stability;

pub use graph::*;
pub use sheaf::*;
pub use flow::*;
pub use spectrum::*;
pub use trajectory::*;
pub use stability::*;

/// Version of the evolving-sheaf library.
pub const VERSION: &str = "0.1.0";
