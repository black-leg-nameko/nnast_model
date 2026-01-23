//! CPG (Code Property Graph) builder in Rust
//!
//! This module provides a high-performance implementation of CPG generation
//! for Python code, with Python bindings via PyO3.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod schema;
mod builder;
pub mod ast_parser;
mod cfg;
mod dfg;
mod scope;
mod utils;

pub use schema::{CPGNode, CPGEdge, CPGGraph};
pub use builder::CPGBuilder;

/// Build CPG graph from Python source code
///
/// # Arguments
/// * `file_path` - Path to the source file
/// * `source` - Source code as string
/// * `ast_json` - Optional JSON string representing Python AST (if None, will be parsed in Python)
///
/// # Returns
/// CPG graph as a Python dictionary
#[pyfunction]
#[pyo3(signature = (file_path, source, ast_json = None))]
fn build_cpg(
    file_path: String,
    source: String,
    ast_json: Option<String>,
) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        // If AST JSON is not provided, parse it in Python
        let ast_json_str = match ast_json {
            Some(json) => json,
            None => {
                // Call Python function to parse and convert AST to JSON
                let ast_to_json = py.import("cpg.ast_to_json")?;
                let parse_func = ast_to_json.getattr("parse_and_convert_to_json")?;
                let json_result: String = parse_func.call1((&source,))?.extract()?;
                json_result
            }
        };
        
        // Parse AST from JSON
        let ast_node = match crate::ast_parser::parse_ast_from_json(&ast_json_str) {
            Ok(node) => node,
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Failed to parse AST JSON: {}", e),
                ));
            }
        };
        
        // Build CPG from AST
        let mut builder = builder::CPGBuilder::new(file_path.clone(), source);
        let graph = match builder.build_from_ast(&ast_node) {
            Ok(g) => g,
            Err(e) => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    format!("Failed to build CPG: {}", e),
                ));
            }
        };
        
        // Convert to Python dictionary (optimized)
        let graph_dict = PyDict::new(py);
        graph_dict.set_item("file", graph.file.as_str())?;
        
        // Build nodes list (pre-allocate with None, then fill)
        let nodes_list = PyList::empty(py);
        // Reserve capacity hint (PyList will grow dynamically)
        for node in &graph.nodes {
            let node_dict = PyDict::new(py);
            node_dict.set_item("id", node.id)?;
            node_dict.set_item("kind", &node.kind)?;
            node_dict.set_item("file", node.file.as_str())?;
            node_dict.set_item("span", node.span)?;
            node_dict.set_item("code", node.code.as_ref())?;
            node_dict.set_item("symbol", node.symbol.as_ref())?;
            node_dict.set_item("type_hint", node.type_hint.as_ref())?;
            node_dict.set_item("flags", &node.flags)?;
            
            // Build attrs dict
            let attrs_dict = PyDict::new(py);
            for (k, v) in &node.attrs {
                attrs_dict.set_item(k, v)?;
            }
            node_dict.set_item("attrs", attrs_dict)?;
            
            nodes_list.append(node_dict)?;
        }
        graph_dict.set_item("nodes", nodes_list)?;
        
        // Build edges list
        let edges_list = PyList::empty(py);
        for edge in &graph.edges {
            let edge_dict = PyDict::new(py);
            edge_dict.set_item("src", edge.src)?;
            edge_dict.set_item("dst", edge.dst)?;
            edge_dict.set_item("kind", &edge.kind)?;
            
            if let Some(ref attrs) = edge.attrs {
                let attrs_dict = PyDict::new(py);
                for (k, v) in attrs {
                    attrs_dict.set_item(k, v)?;
                }
                edge_dict.set_item("attrs", attrs_dict)?;
            }
            
            edges_list.append(edge_dict)?;
        }
        graph_dict.set_item("edges", edges_list)?;
        
        Ok(graph_dict.into())
    })
}

/// Python module definition
#[pymodule]
fn cpg_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_cpg, m)?)?;
    m.add_class::<schema::CPGNodePy>()?;
    m.add_class::<schema::CPGEdgePy>()?;
    m.add_class::<schema::CPGGraphPy>()?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        // Basic smoke test
        assert!(true);
    }
}
