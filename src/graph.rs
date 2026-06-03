

/// A directed edge in the graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    pub weight: f64,
}

impl Edge {
    pub fn new(src: usize, dst: usize, weight: f64) -> Self {
        Edge { src, dst, weight }
    }
}

/// A directed graph with vertex-weighted edges.
#[derive(Debug, Clone)]
pub struct Graph {
    pub n_vertices: usize,
    pub n_edges: usize,
    pub edges: Vec<Edge>,
}

impl Graph {
    pub fn new(n_vertices: usize, edges: Vec<Edge>) -> Self {
        let n_edges = edges.len();
        Graph {
            n_vertices,
            n_edges,
            edges,
        }
    }

    /// Build a cycle graph C_n.
    pub fn cycle(n: usize) -> Self {
        let edges: Vec<Edge> = (0..n)
            .map(|i| Edge::new(i, (i + 1) % n, 1.0))
            .collect();
        Graph::new(n, edges)
    }

    /// Build a path graph P_n.
    pub fn path(n: usize) -> Self {
        assert!(n >= 2, "path graph needs at least 2 vertices");
        let edges: Vec<Edge> = (0..n - 1)
            .map(|i| Edge::new(i, i + 1, 1.0))
            .collect();
        Graph::new(n, edges)
    }

    /// Build a complete graph K_n.
    pub fn complete(n: usize) -> Self {
        let mut edges = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push(Edge::new(i, j, 1.0));
            }
        }
        Graph::new(n, edges)
    }

    /// Build a 3-regular expander-like graph (Paley-ish construction).
    pub fn expander(n: usize) -> Self {
        let max_edges = n * 3;
        let mut edges = Vec::with_capacity(max_edges);
        for i in 0..n {
            let targets = [(i + 1) % n, (i + 2) % n, (i + n / 3) % n];
            for &t in &targets {
                let (a, b) = if i < t { (i, t) } else { (t, i) };
                if !edges.iter().any(|e: &Edge| e.src == a && e.dst == b) {
                    edges.push(Edge::new(a, b, 1.0));
                }
            }
        }
        Graph::new(n, edges)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycle_4_vertices() {
        let g = Graph::cycle(4);
        assert_eq!(g.n_vertices, 4);
        assert_eq!(g.n_edges, 4);
        for i in 0..4 {
            assert_eq!(g.edges[i].src, i);
            assert_eq!(g.edges[i].dst, (i + 1) % 4);
        }
    }

    #[test]
    fn test_path_5() {
        let g = Graph::path(5);
        assert_eq!(g.n_vertices, 5);
        assert_eq!(g.n_edges, 4);
    }

    #[test]
    fn test_complete_k4() {
        let g = Graph::complete(4);
        assert_eq!(g.n_vertices, 4);
        assert_eq!(g.n_edges, 6);
    }

    #[test]
    fn test_expander_10() {
        let g = Graph::expander(10);
        assert_eq!(g.n_vertices, 10);
        assert!(g.n_edges >= 10);
    }

    #[test]
    fn test_cycle_3() {
        let g = Graph::cycle(3);
        assert_eq!(g.n_vertices, 3);
        assert_eq!(g.n_edges, 3);
    }

    #[test]
    fn test_new_graph() {
        let edges = vec![Edge::new(0, 1, 2.0)];
        let g = Graph::new(2, edges);
        assert_eq!(g.n_vertices, 2);
        assert_eq!(g.n_edges, 1);
        assert!((g.edges[0].weight - 2.0).abs() < 1e-12);
    }
}
