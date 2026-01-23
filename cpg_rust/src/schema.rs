//! CPG data structures

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// CPG Node representing an AST node or statement
#[derive(Debug, Clone)]
pub struct CPGNode {
    pub id: u32,
    pub kind: String,
    pub file: Arc<String>,  // Use Arc to avoid cloning for each node
    pub span: (u32, u32, u32, u32), // (start_line, start_col, end_line, end_col)
    pub code: Option<String>,
    pub symbol: Option<String>,
    pub type_hint: Option<String>,
    pub flags: Vec<String>,
    pub attrs: HashMap<String, String>,
}

// Custom serialization for CPGNode (Arc<String> -> String)
impl Serialize for CPGNode {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("CPGNode", 9)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("kind", &self.kind)?;
        state.serialize_field("file", self.file.as_str())?;
        state.serialize_field("span", &self.span)?;
        state.serialize_field("code", &self.code)?;
        state.serialize_field("symbol", &self.symbol)?;
        state.serialize_field("type_hint", &self.type_hint)?;
        state.serialize_field("flags", &self.flags)?;
        state.serialize_field("attrs", &self.attrs)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for CPGNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Id,
            Kind,
            File,
            Span,
            Code,
            Symbol,
            TypeHint,
            Flags,
            Attrs,
        }

        struct CPGNodeVisitor;

        impl<'de> Visitor<'de> for CPGNodeVisitor {
            type Value = CPGNode;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CPGNode")
            }

            fn visit_map<V>(self, mut map: V) -> Result<CPGNode, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut id = None;
                let mut kind = None;
                let mut file = None;
                let mut span = None;
                let mut code = None;
                let mut symbol = None;
                let mut type_hint = None;
                let mut flags = None;
                let mut attrs = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Id => {
                            if id.is_some() {
                                return Err(de::Error::duplicate_field("id"));
                            }
                            id = Some(map.next_value()?);
                        }
                        Field::Kind => {
                            if kind.is_some() {
                                return Err(de::Error::duplicate_field("kind"));
                            }
                            kind = Some(map.next_value()?);
                        }
                        Field::File => {
                            if file.is_some() {
                                return Err(de::Error::duplicate_field("file"));
                            }
                            let file_str: String = map.next_value()?;
                            file = Some(Arc::new(file_str));
                        }
                        Field::Span => {
                            if span.is_some() {
                                return Err(de::Error::duplicate_field("span"));
                            }
                            span = Some(map.next_value()?);
                        }
                        Field::Code => {
                            if code.is_some() {
                                return Err(de::Error::duplicate_field("code"));
                            }
                            code = Some(map.next_value()?);
                        }
                        Field::Symbol => {
                            if symbol.is_some() {
                                return Err(de::Error::duplicate_field("symbol"));
                            }
                            symbol = Some(map.next_value()?);
                        }
                        Field::TypeHint => {
                            if type_hint.is_some() {
                                return Err(de::Error::duplicate_field("type_hint"));
                            }
                            type_hint = Some(map.next_value()?);
                        }
                        Field::Flags => {
                            if flags.is_some() {
                                return Err(de::Error::duplicate_field("flags"));
                            }
                            flags = Some(map.next_value()?);
                        }
                        Field::Attrs => {
                            if attrs.is_some() {
                                return Err(de::Error::duplicate_field("attrs"));
                            }
                            attrs = Some(map.next_value()?);
                        }
                    }
                }

                let id = id.ok_or_else(|| de::Error::missing_field("id"))?;
                let kind = kind.ok_or_else(|| de::Error::missing_field("kind"))?;
                let file = file.ok_or_else(|| de::Error::missing_field("file"))?;
                let span = span.ok_or_else(|| de::Error::missing_field("span"))?;
                let code = code.unwrap_or(None);
                let symbol = symbol.unwrap_or(None);
                let type_hint = type_hint.unwrap_or(None);
                let flags = flags.unwrap_or_default();
                let attrs = attrs.unwrap_or_default();

                Ok(CPGNode {
                    id,
                    kind,
                    file,
                    span,
                    code,
                    symbol,
                    type_hint,
                    flags,
                    attrs,
                })
            }
        }

        const FIELDS: &[&str] = &["id", "kind", "file", "span", "code", "symbol", "type_hint", "flags", "attrs"];
        deserializer.deserialize_struct("CPGNode", FIELDS, CPGNodeVisitor)
    }
}

/// CPG Edge representing relationships between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CPGEdge {
    pub src: u32,
    pub dst: u32,
    pub kind: String, // "AST", "CFG", "DFG", "DDFG"
    pub attrs: Option<HashMap<String, String>>,
}

/// Complete CPG Graph
#[derive(Debug, Clone)]
pub struct CPGGraph {
    pub file: Arc<String>,  // Use Arc to avoid cloning
    pub nodes: Vec<CPGNode>,
    pub edges: Vec<CPGEdge>,
}

// Custom serialization for CPGGraph
impl Serialize for CPGGraph {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("CPGGraph", 3)?;
        state.serialize_field("file", self.file.as_str())?;
        state.serialize_field("nodes", &self.nodes)?;
        state.serialize_field("edges", &self.edges)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for CPGGraph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            File,
            Nodes,
            Edges,
        }

        struct CPGGraphVisitor;

        impl<'de> Visitor<'de> for CPGGraphVisitor {
            type Value = CPGGraph;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CPGGraph")
            }

            fn visit_map<V>(self, mut map: V) -> Result<CPGGraph, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut file = None;
                let mut nodes = None;
                let mut edges = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::File => {
                            if file.is_some() {
                                return Err(de::Error::duplicate_field("file"));
                            }
                            let file_str: String = map.next_value()?;
                            file = Some(Arc::new(file_str));
                        }
                        Field::Nodes => {
                            if nodes.is_some() {
                                return Err(de::Error::duplicate_field("nodes"));
                            }
                            nodes = Some(map.next_value()?);
                        }
                        Field::Edges => {
                            if edges.is_some() {
                                return Err(de::Error::duplicate_field("edges"));
                            }
                            edges = Some(map.next_value()?);
                        }
                    }
                }

                let file = file.ok_or_else(|| de::Error::missing_field("file"))?;
                let nodes = nodes.ok_or_else(|| de::Error::missing_field("nodes"))?;
                let edges = edges.ok_or_else(|| de::Error::missing_field("edges"))?;

                Ok(CPGGraph { file, nodes, edges })
            }
        }

        const FIELDS: &[&str] = &["file", "nodes", "edges"];
        deserializer.deserialize_struct("CPGGraph", FIELDS, CPGGraphVisitor)
    }
}

/// Python binding for CPGNode
#[pyclass]
#[derive(Clone)]
pub struct CPGNodePy {
    #[pyo3(get, set)]
    pub id: u32,
    #[pyo3(get, set)]
    pub kind: String,
    #[pyo3(get, set)]
    pub file: String,
    #[pyo3(get, set)]
    pub span: (u32, u32, u32, u32),
    #[pyo3(get, set)]
    pub code: Option<String>,
    #[pyo3(get, set)]
    pub symbol: Option<String>,
    #[pyo3(get, set)]
    pub type_hint: Option<String>,
    #[pyo3(get, set)]
    pub flags: Vec<String>,
    #[pyo3(get, set)]
    pub attrs: HashMap<String, String>,
}

#[pymethods]
impl CPGNodePy {
    #[new]
    #[pyo3(signature = (id, kind, file, span, code = None, symbol = None, type_hint = None, flags = None, attrs = None))]
    fn new(
        id: u32,
        kind: String,
        file: String,
        span: (u32, u32, u32, u32),
        code: Option<String>,
        symbol: Option<String>,
        type_hint: Option<String>,
        flags: Option<Vec<String>>,
        attrs: Option<HashMap<String, String>>,
    ) -> Self {
        Self {
            id,
            kind,
            file,
            span,
            code,
            symbol,
            type_hint,
            flags: flags.unwrap_or_default(),
            attrs: attrs.unwrap_or_default(),
        }
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("id", self.id)?;
        dict.set_item("kind", &self.kind)?;
        dict.set_item("file", &self.file)?;
        dict.set_item("span", self.span)?;
        dict.set_item("code", self.code.as_ref())?;
        dict.set_item("symbol", self.symbol.as_ref())?;
        dict.set_item("type_hint", self.type_hint.as_ref())?;
        dict.set_item("flags", &self.flags)?;
        dict.set_item("attrs", &self.attrs)?;
        Ok(dict.into())
    }
}

impl From<CPGNode> for CPGNodePy {
    fn from(node: CPGNode) -> Self {
        Self {
            id: node.id,
            kind: node.kind,
            file: (*node.file).clone(),  // Arc<String> -> String
            span: node.span,
            code: node.code,
            symbol: node.symbol,
            type_hint: node.type_hint,
            flags: node.flags,
            attrs: node.attrs,
        }
    }
}

impl From<CPGNodePy> for CPGNode {
    fn from(node: CPGNodePy) -> Self {
        Self {
            id: node.id,
            kind: node.kind,
            file: Arc::new(node.file),  // String -> Arc<String>
            span: node.span,
            code: node.code,
            symbol: node.symbol,
            type_hint: node.type_hint,
            flags: node.flags,
            attrs: node.attrs,
        }
    }
}

/// Python binding for CPGEdge
#[pyclass]
#[derive(Clone)]
pub struct CPGEdgePy {
    #[pyo3(get, set)]
    pub src: u32,
    #[pyo3(get, set)]
    pub dst: u32,
    #[pyo3(get, set)]
    pub kind: String,
    #[pyo3(get, set)]
    pub attrs: Option<HashMap<String, String>>,
}

#[pymethods]
impl CPGEdgePy {
    #[new]
    fn new(
        src: u32,
        dst: u32,
        kind: String,
        attrs: Option<HashMap<String, String>>,
    ) -> Self {
        Self { src, dst, kind, attrs }
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("src", self.src)?;
        dict.set_item("dst", self.dst)?;
        dict.set_item("kind", &self.kind)?;
        dict.set_item("attrs", self.attrs.as_ref())?;
        Ok(dict.into())
    }
}

impl From<CPGEdge> for CPGEdgePy {
    fn from(edge: CPGEdge) -> Self {
        Self {
            src: edge.src,
            dst: edge.dst,
            kind: edge.kind,
            attrs: edge.attrs,
        }
    }
}

impl From<CPGEdgePy> for CPGEdge {
    fn from(edge: CPGEdgePy) -> Self {
        Self {
            src: edge.src,
            dst: edge.dst,
            kind: edge.kind,
            attrs: edge.attrs,
        }
    }
}

/// Python binding for CPGGraph
#[pyclass]
#[derive(Clone)]
pub struct CPGGraphPy {
    #[pyo3(get, set)]
    pub file: String,
    #[pyo3(get, set)]
    pub nodes: Vec<CPGNodePy>,
    #[pyo3(get, set)]
    pub edges: Vec<CPGEdgePy>,
}

#[pymethods]
impl CPGGraphPy {
    #[new]
    fn new(file: String, nodes: Vec<CPGNodePy>, edges: Vec<CPGEdgePy>) -> Self {
        Self { file, nodes, edges }
    }

    fn to_dict(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("file", &self.file)?;
        
        let nodes_list = PyList::empty(py);
        for node in &self.nodes {
            nodes_list.append(node.to_dict(py)?)?;
        }
        dict.set_item("nodes", nodes_list)?;
        
        let edges_list = PyList::empty(py);
        for edge in &self.edges {
            edges_list.append(edge.to_dict(py)?)?;
        }
        dict.set_item("edges", edges_list)?;
        
        Ok(dict.into())
    }
}

impl From<CPGGraph> for CPGGraphPy {
    fn from(graph: CPGGraph) -> Self {
        Self {
            file: (*graph.file).clone(),  // Arc<String> -> String
            nodes: graph.nodes.into_iter().map(CPGNodePy::from).collect(),
            edges: graph.edges.into_iter().map(CPGEdgePy::from).collect(),
        }
    }
}

impl From<CPGGraphPy> for CPGGraph {
    fn from(graph: CPGGraphPy) -> Self {
        Self {
            file: Arc::new(graph.file),  // String -> Arc<String>
            nodes: graph.nodes.into_iter().map(CPGNode::from).collect(),
            edges: graph.edges.into_iter().map(CPGEdge::from).collect(),
        }
    }
}
