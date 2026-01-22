//! CPG Builder - Main logic for building Code Property Graphs

use crate::schema::{CPGNode, CPGEdge, CPGGraph};
use crate::scope::ScopeManager;
use crate::cfg::CFGBuilder;
use crate::dfg::DFGBuilder;
use crate::ast_parser::{ASTNode, node_type_to_kind, extract_symbol, extract_type_hint, 
                        is_statement, defines_scope};
use crate::utils::extract_code;
use std::collections::HashMap;
use std::sync::Arc;

/// Main CPG builder
pub struct CPGBuilder {
    file_path: Arc<String>,  // Use Arc to avoid cloning
    #[allow(dead_code)]  // Will be used for code extraction
    source: String,
    source_lines: Vec<String>,
    nodes: Vec<CPGNode>,
    edges: Vec<CPGEdge>,
    next_id: u32,
    scope_manager: ScopeManager,
    #[allow(dead_code)]  // Will be used in Phase 3 (CFG construction)
    cfg_builder: CFGBuilder,
    #[allow(dead_code)]  // Will be used in Phase 4 (DFG construction)
    dfg_builder: DFGBuilder,
    // For CFG construction
    // Using a simple counter-based approach instead of pointers for safety
    ast_node_counter: usize,
    last_stmt_by_parent: HashMap<usize, u32>,  // parent counter -> last stmt node id
    last_stmt_by_parent_field: HashMap<(usize, String), u32>,  // (parent counter, field_name) -> last stmt node id
    node_ids: HashMap<usize, u32>,  // AST node counter -> CPG node id
    child_field_names: HashMap<usize, String>,  // child counter -> field name (body, orelse, etc.)
    parent_children: HashMap<usize, Vec<usize>>,  // parent counter -> Vec<child counter>
    node_types: HashMap<usize, String>,  // node counter -> AST node type
    comprehension_stack: Vec<u32>,  // Stack of comprehension node IDs (for DFG edges)
}

impl CPGBuilder {
    /// Create a new CPG builder
    pub fn new(file_path: String, source: String) -> Self {
        let source_lines: Vec<String> = source.lines().map(|s| s.to_string()).collect();
        
        // 予想されるノード数を基に容量を事前確保（パフォーマンス最適化）
        let estimated_nodes = source_lines.len() * 3; // 大まかな見積もり
        
        Self {
            file_path: Arc::new(file_path),
            source,
            source_lines,
            nodes: Vec::with_capacity(estimated_nodes),
            edges: Vec::with_capacity(estimated_nodes * 2),
            next_id: 1,
            scope_manager: ScopeManager::new(),
            cfg_builder: CFGBuilder::new(),
            dfg_builder: DFGBuilder::new(),
            ast_node_counter: 0,
            last_stmt_by_parent: HashMap::with_capacity(estimated_nodes / 10),
            last_stmt_by_parent_field: HashMap::with_capacity(estimated_nodes / 10),
            node_ids: HashMap::with_capacity(estimated_nodes),
            child_field_names: HashMap::with_capacity(estimated_nodes),
            parent_children: HashMap::with_capacity(estimated_nodes / 10),
            node_types: HashMap::with_capacity(estimated_nodes),
            comprehension_stack: Vec::new(),
        }
    }

    /// Build CPG graph from AST node
    pub fn build_from_ast(&mut self, ast_node: &ASTNode) -> Result<CPGGraph, Box<dyn std::error::Error>> {
        // Traverse AST and build CPG
        self.visit_ast_node(ast_node, None, None)?;
        
        Ok(CPGGraph {
            file: Arc::clone(&self.file_path),
            nodes: std::mem::take(&mut self.nodes),
            edges: std::mem::take(&mut self.edges),
        })
    }

    /// Visit AST node and build CPG nodes/edges
    fn visit_ast_node(
        &mut self,
        node: &ASTNode,
        parent_counter: Option<usize>,
        field_name: Option<&str>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let node_counter = self.ast_node_counter;
        self.ast_node_counter += 1;
        
        // Store field name mapping for CFG construction
        if let Some(field_name) = field_name {
            self.child_field_names.insert(node_counter, field_name.to_string());
        }
        
        // Store parent-child relationship
        if let Some(parent_cnt) = parent_counter {
            self.parent_children
                .entry(parent_cnt)
                .or_insert_with(Vec::new)
                .push(node_counter);
        }
        // Calculate span
        let span = match node.span {
            Some(s) => s,
            None => {
                // For nodes without positions, we still need to process them
                // but we won't create a CPG node. However, we need to track the counter
                // so that children can reference this parent.
                // Store a placeholder mapping (using 0 as a sentinel value)
                // Actually, let's create a node anyway but mark it as having no span
                // This ensures parent-child relationships are maintained
                
                // Generate a node ID even for nodes without span
                let node_id = self.next_id();
                
                // Create a minimal CPG node
                let cpg_node = CPGNode {
                    id: node_id,
                    kind: node_type_to_kind(&node.node_type).to_string(),
                    file: Arc::clone(&self.file_path),
                    span: (0, 0, 0, 0), // Placeholder span
                    code: None,
                    symbol: extract_symbol(node),
                    type_hint: extract_type_hint(node),
                    flags: Vec::new(),
                    attrs: node.attrs.clone(),
                };
                
                self.nodes.push(cpg_node);
                self.node_ids.insert(node_counter, node_id);
                self.node_types.insert(node_counter, node.node_type.clone());
                
                // Add AST edge to parent if exists
                if let Some(parent_cnt) = parent_counter {
                    if let Some(&parent_node_id) = self.node_ids.get(&parent_cnt) {
                        self.edges.push(CPGEdge {
                            src: parent_node_id,
                            dst: node_id,
                            kind: "AST".to_string(),
                            attrs: None,
                        });
                    }
                }
                
                // Traverse children
                for child in &node.children {
                    let child_field_name = child.attrs.get("_field_name").map(|s| s.as_str());
                    self.visit_ast_node(child, Some(node_counter), child_field_name)?;
                }
                
                return Ok(node_counter);
            }
        };

        // Generate node ID
        let node_id = self.next_id();
        
        // Get kind
        let kind = node_type_to_kind(&node.node_type).to_string();
        
        // Extract code snippet
        let code = extract_code(&self.source_lines, span);
        
        // Extract symbol
        let symbol = extract_symbol(node);
        
        // Extract type hint
        let type_hint = extract_type_hint(node);
        
        // Build attributes
        let mut attrs = node.attrs.clone();
        if let Some(ref th) = type_hint {
            // Parse container types like List[int]
            if th.contains('[') && th.ends_with(']') {
                if let Some(bracket_pos) = th.find('[') {
                    let container = th[..bracket_pos].trim();
                    let inner = th[bracket_pos + 1..th.len() - 1].trim();
                    attrs.insert("DataType".to_string(), inner.to_string());
                    attrs.insert("ContainerType".to_string(), container.to_string());
                } else {
                    attrs.insert("DataType".to_string(), th.clone());
                }
            } else {
                attrs.insert("DataType".to_string(), th.clone());
            }
        }
        
        // Extract symbol before creating CPG node (needed for scope management)
        let symbol_for_scope = symbol.clone();
        
        // Create CPG node
        let cpg_node = CPGNode {
            id: node_id,
            kind,
            file: self.file_path.clone(),
            span,
            code,
            symbol,
            type_hint,
            flags: Vec::new(),
            attrs,
        };
        
        self.nodes.push(cpg_node);
        
        // Store mapping for CFG construction (AFTER node is created)
        self.node_ids.insert(node_counter, node_id);
        self.node_types.insert(node_counter, node.node_type.clone());
        
        // Add AST edge to parent (if exists)
        // This creates the tree structure
        // Note: parent_counter is the AST node counter, not the CPG node ID
        if let Some(parent_cnt) = parent_counter {
            // Parent node should already be in node_ids since we visit parent before children
            if let Some(&parent_node_id) = self.node_ids.get(&parent_cnt) {
                self.edges.push(CPGEdge {
                    src: parent_node_id,
                    dst: node_id,
                    kind: "AST".to_string(),
                    attrs: None,
                });
            }
            // If parent is not found, it might be because parent has no span (was skipped)
            // This is okay - we just don't add an AST edge in that case
        }
        
        // Handle CFG edges for statements within the same block
        // CFG edges connect statements sequentially within a block
        // Python: if isinstance(node, ast.stmt) and parent_ast is not None
        // In Python, ALL statements get CFG edges if they have a parent
        // The parent doesn't need to be a specific type - any parent works
        // 
        // IMPORTANT: In Python, ast.stmt includes FunctionDef, ClassDef, etc.
        // Python implementation: if isinstance(node, ast.stmt) and parent_ast is not None
        // This means ALL statements (including FunctionDef) get tracked in last_stmt_by_parent
        // if they have a parent. FunctionDef itself becomes the "last statement" for its parent (Module),
        // and then body statements use FunctionDef as their parent.
        // 
        // However, for control structures (If, For, While, etc.), body and orelse are separate blocks.
        // We should NOT create sequential CFG edges between body and orelse.
        // Instead, we track them separately by using (parent_counter, field_name) as key.
        if is_statement(&node.node_type) {
            if let Some(parent_cnt) = parent_counter {
                // Check if this statement is in a control structure's body/orelse
                let field_name = node.attrs.get("_field_name").map(|s| s.as_str());
                let parent_node_type = self.node_types.get(&parent_cnt);
                let parent_is_control = parent_node_type.map(|t| {
                    matches!(t.as_str(), "If" | "For" | "While" | "AsyncFor" | "Try" | "With" | "AsyncWith")
                }).unwrap_or(false);
                
                if parent_is_control && field_name.is_some() {
                    // For control structure blocks, track by (parent_counter, field_name)
                    let field_name_str = field_name.unwrap().to_string();
                    let key = (parent_cnt, field_name_str.clone());
                    
                    if let Some(&last_stmt_id) = self.last_stmt_by_parent_field.get(&key) {
                        // Add sequential CFG edge from previous statement to current within the same block
                        self.edges.push(CPGEdge {
                            src: last_stmt_id,
                            dst: node_id,
                            kind: "CFG".to_string(),
                            attrs: None,
                        });
                    }
                    // Update the last statement for this (parent, field) pair
                    self.last_stmt_by_parent_field.insert(key, node_id);
                } else {
                    // Normal sequential flow within the same block
                if let Some(&last_stmt_id) = self.last_stmt_by_parent.get(&parent_cnt) {
                    // Add sequential CFG edge from previous statement to current
                    self.edges.push(CPGEdge {
                        src: last_stmt_id,
                        dst: node_id,
                        kind: "CFG".to_string(),
                        attrs: None,
                    });
                }
                // Update the last statement for this parent
                // This tracks the last statement in each parent's block
                // Python implementation: self._last_stmt_by_parent[parent_ast] = node_id
                self.last_stmt_by_parent.insert(parent_cnt, node_id);
                }
            }
            // Note: If parent_counter is None, this is the root Module node
            // Module itself is a statement but has no parent, so no CFG edge is added
        }
        
        // Special case: For FunctionDef/ClassDef nodes, we need to track them for their body statements
        // Even if parent_counter is None (Module), we should track FunctionDef/ClassDef itself
        // so that body statements can use FunctionDef/ClassDef as their parent
        if matches!(node.node_type.as_str(), "FunctionDef" | "AsyncFunctionDef" | "ClassDef") {
            // Track this node itself so body statements can reference it
            // This ensures that body statements can find FunctionDef/ClassDef as their parent
            self.last_stmt_by_parent.insert(node_counter, node_id);
        }
        
        // --- DFG: variables, attributes, and simple calls ---
        // Definitions:
        //   - function arguments (ast.arg)
        //   - names in Store context
        // Uses:
        //   - names in Load / Del context
        //   - attributes (obj.attr) as uses of the base object
        if node.node_type == "arg" {
            if let Some(ref symbol_name) = symbol_for_scope {
                self.scope_manager.define(symbol_name.clone(), node_id);
            }
        } else if node.node_type == "Name" {
            if let Some(ref symbol_name) = symbol_for_scope {
                let ctx = node.ctx.as_ref().map(|s| s.as_str());
                if ctx == Some("Store") {
                    // Definition
                    self.scope_manager.define(symbol_name.clone(), node_id);
                } else {
                    // Use: Load or Del context
                    if let Some(def_id) = self.scope_manager.resolve(symbol_name) {
                        self.edges.push(CPGEdge {
                            src: def_id,
                            dst: node_id,
                            kind: "DFG".to_string(),
                            attrs: None,
                        });
                    }
                }
            }
        } else if node.node_type == "Attribute" {
            // obj.attr: create DFG edge from obj definition to this attribute node
            // when obj is a simple name (Name node)
            // Find the value child (which should be the base object)
            let value_children: Vec<&ASTNode> = node.children
                .iter()
                .filter(|child| {
                    child.attrs.get("_field_name")
                        .map(|s| s.as_str())
                        .unwrap_or("") == "value"
                })
                .collect();
            
            if let Some(value_node) = value_children.first() {
                if value_node.node_type == "Name" {
                    if let Some(base_name) = &value_node.symbol {
                        if let Some(def_id) = self.scope_manager.resolve(base_name) {
                            self.edges.push(CPGEdge {
                                src: def_id,
                                dst: node_id,
                                kind: "DFG".to_string(),
                                attrs: None,
                            });
                        }
                    }
                }
            }
        } else if node.node_type == "Call" {
            // For function calls like f(x, y), connect argument definitions
            // to their usage sites inside the call node
            // Find args children
            let args_children: Vec<&ASTNode> = node.children
                .iter()
                .filter(|child| {
                    child.attrs.get("_field_name")
                        .map(|s| s.as_str())
                        .unwrap_or("") == "args"
                })
                .collect();
            
            for arg_node in args_children {
                if arg_node.node_type == "Name" {
                    if let Some(arg_name) = &arg_node.symbol {
                        if let Some(def_id) = self.scope_manager.resolve(arg_name) {
                            self.edges.push(CPGEdge {
                                src: def_id,
                                dst: node_id,
                                kind: "DFG".to_string(),
                                attrs: None,
                            });
                        }
                    }
                }
            }
            
            // For method calls like xs.count(0), also connect the base object
            // definition to the call node
            // Find func child (which should be an Attribute)
            let func_children: Vec<&ASTNode> = node.children
                .iter()
                .filter(|child| {
                    child.attrs.get("_field_name")
                        .map(|s| s.as_str())
                        .unwrap_or("") == "func"
                })
                .collect();
            
            if let Some(func_node) = func_children.first() {
                if func_node.node_type == "Attribute" {
                    // Find the value child of the Attribute (the base object)
                    let func_value_children: Vec<&ASTNode> = func_node.children
                        .iter()
                        .filter(|child| {
                            child.attrs.get("_field_name")
                                .map(|s| s.as_str())
                                .unwrap_or("") == "value"
                        })
                        .collect();
                    
                    if let Some(func_value_node) = func_value_children.first() {
                        if func_value_node.node_type == "Name" {
                            if let Some(base_name) = &func_value_node.symbol {
                                if let Some(def_id) = self.scope_manager.resolve(base_name) {
                                    self.edges.push(CPGEdge {
                                        src: def_id,
                                        dst: node_id,
                                        kind: "DFG".to_string(),
                                        attrs: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        } else if matches!(node.node_type.as_str(), "ListComp" | "SetComp" | "DictComp" | "GeneratorExp") {
            // Push comprehension node onto stack so children can reference it
            self.comprehension_stack.push(node_id);
        }
        
        // For Name nodes inside comprehensions, also create DFG edge to comprehension node
        if node.node_type == "Name" && !self.comprehension_stack.is_empty() {
            let ctx = node.ctx.as_ref().map(|s| s.as_str());
            if ctx != Some("Store") {
                // This is a use (Load or Del context) inside a comprehension
                if let Some(ref symbol_name) = symbol_for_scope {
                    if let Some(def_id) = self.scope_manager.resolve(symbol_name) {
                        // Connect to comprehension node
                        if let Some(&comprehension_id) = self.comprehension_stack.last() {
                            self.edges.push(CPGEdge {
                                src: def_id,
                                dst: comprehension_id,
                                kind: "DFG".to_string(),
                                attrs: None,
                            });
                        }
                    }
                }
            }
        }
        
        // Handle scope management
        if defines_scope(&node.node_type) {
            self.scope_manager.enter_scope();
        }
        
        // Define symbols (for scope management, but DFG definitions are handled above)
        // Note: arg and Name(Store) are already handled in DFG section above
        if let Some(ref symbol_name) = symbol_for_scope {
            if node.node_type == "FunctionDef" ||
               node.node_type == "AsyncFunctionDef" ||
               node.node_type == "ClassDef" {
                self.scope_manager.define(symbol_name.clone(), node_id);
            }
        }
        
        // Recursively visit children
        for child in &node.children {
            let child_field_name = child.attrs.get("_field_name").map(|s| s.as_str());
            self.visit_ast_node(child, Some(node_counter), child_field_name)?;
        }
        
        // --- Control-structure-aware CFG edges (after children are visited) ---
        // This matches the Python implementation's logic
        self.build_control_structure_cfg_edges(node, node_id, node_counter)?;
        
        // Exit scope if needed
        if defines_scope(&node.node_type) {
            self.scope_manager.exit_scope();
        }
        
        // Pop comprehension stack if this was a comprehension node
        if matches!(node.node_type.as_str(), "ListComp" | "SetComp" | "DictComp" | "GeneratorExp") {
            self.comprehension_stack.pop();
        }
        
        Ok(node_counter)
    }

    /// Generate next node ID
    fn next_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Build CFG edges for control structures (after children are visited)
    fn build_control_structure_cfg_edges(
        &mut self,
        node: &ASTNode,
        head_id: u32,
        node_counter: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match node.node_type.as_str() {
            "If" => {
                // Find body and orelse children by field name
                let body_counters = self.find_child_counters_by_field(node_counter, "body");
                let orelse_counters = self.find_child_counters_by_field(node_counter, "orelse");
                
                // Connect head to first statement in then-branch
                if let Some(first_body_id) = self.get_first_stmt_id_from_counters(&body_counters) {
                    self.edges.push(CPGEdge {
                        src: head_id,
                        dst: first_body_id,
                        kind: "CFG".to_string(),
                        attrs: None,
                    });
                }
                
                // Connect head to first statement in else-branch
                if let Some(first_else_id) = self.get_first_stmt_id_from_counters(&orelse_counters) {
                    self.edges.push(CPGEdge {
                        src: head_id,
                        dst: first_else_id,
                        kind: "CFG".to_string(),
                        attrs: None,
                    });
                }
            }
            "For" | "While" | "AsyncFor" => {
                // Find body children
                let body_counters = self.find_child_counters_by_field(node_counter, "body");
                
                // Connect head to first statement in body
                if let Some(first_body_id) = self.get_first_stmt_id_from_counters(&body_counters) {
                    self.edges.push(CPGEdge {
                        src: head_id,
                        dst: first_body_id,
                        kind: "CFG".to_string(),
                        attrs: None,
                    });
                }
                
                // Back edge from last statement in body to head
                if let Some(last_body_id) = self.get_last_stmt_id_from_counters(&body_counters) {
                    self.edges.push(CPGEdge {
                        src: last_body_id,
                        dst: head_id,
                        kind: "CFG".to_string(),
                        attrs: None,
                    });
                }
            }
            "Try" => {
                // Find body, handlers, and finalbody children
                let body_counters = self.find_child_counters_by_field(node_counter, "body");
                let handlers_counters = self.find_child_counters_by_field(node_counter, "handlers");
                let finalbody_counters = self.find_child_counters_by_field(node_counter, "finalbody");
                
                // Connect head to first statement in try-body
                if let Some(first_body_id) = self.get_first_stmt_id_from_counters(&body_counters) {
                    self.edges.push(CPGEdge {
                        src: head_id,
                        dst: first_body_id,
                        kind: "CFG".to_string(),
                        attrs: None,
                    });
                }
                
                // Connect head to each except-handler (first statement in handler)
                for handler_counter in &handlers_counters {
                    if let Some(&handler_node_id) = self.node_ids.get(handler_counter) {
                        // Find first statement in handler body
                        let handler_body_counters = self.find_child_counters_by_field(*handler_counter, "body");
                        if let Some(first_handler_body_id) = self.get_first_stmt_id_from_counters(&handler_body_counters) {
                            self.edges.push(CPGEdge {
                                src: head_id,
                                dst: first_handler_body_id,
                                kind: "CFG".to_string(),
                                attrs: None,
                            });
                        } else {
                            // If handler has no body statements, connect to handler node itself
                            self.edges.push(CPGEdge {
                                src: head_id,
                                dst: handler_node_id,
                                kind: "CFG".to_string(),
                                attrs: None,
                            });
                        }
                    }
                }
                
                // Connect head to first statement in finalbody
                if let Some(first_final_id) = self.get_first_stmt_id_from_counters(&finalbody_counters) {
                    self.edges.push(CPGEdge {
                        src: head_id,
                        dst: first_final_id,
                        kind: "CFG".to_string(),
                        attrs: None,
                    });
                }
            }
            "With" | "AsyncWith" => {
                // Find body children
                let body_counters = self.find_child_counters_by_field(node_counter, "body");
                
                // Connect head to first statement in body
                if let Some(first_body_id) = self.get_first_stmt_id_from_counters(&body_counters) {
                    self.edges.push(CPGEdge {
                        src: head_id,
                        dst: first_body_id,
                        kind: "CFG".to_string(),
                        attrs: None,
                    });
                }
            }
            "ListComp" | "SetComp" | "GeneratorExp" => {
                // Find elt child
                let elt_counters = self.find_child_counters_by_field(node_counter, "elt");
                if let Some(elt_id) = self.get_first_stmt_id_from_counters(&elt_counters) {
                    self.edges.push(CPGEdge {
                        src: head_id,
                        dst: elt_id,
                        kind: "CFG".to_string(),
                        attrs: None,
                    });
                }
            }
            "DictComp" => {
                // Find value child
                let value_counters = self.find_child_counters_by_field(node_counter, "value");
                if let Some(value_id) = self.get_first_stmt_id_from_counters(&value_counters) {
                    self.edges.push(CPGEdge {
                        src: head_id,
                        dst: value_id,
                        kind: "CFG".to_string(),
                        attrs: None,
                    });
                }
            }
            _ => {
                // Not a control structure, no special CFG edges
            }
        }

        Ok(())
    }

    /// Find child node counters by field name for a given parent
    fn find_child_counters_by_field(&self, parent_counter: usize, field_name: &str) -> Vec<usize> {
        // Get all children of this parent
        let children = match self.parent_children.get(&parent_counter) {
            Some(children) => children,
            None => return Vec::new(),
        };
        
        // Filter by field name
        children
            .iter()
            .filter(|&&child_counter| {
                self.child_field_names
                    .get(&child_counter)
                    .map(|s| s.as_str())
                    .unwrap_or("") == field_name
            })
            .copied()
            .collect()
    }

    /// Get first statement node ID from child counters
    fn get_first_stmt_id_from_counters(&self, child_counters: &[usize]) -> Option<u32> {
        use crate::ast_parser::is_statement;
        
        // Find the first child counter that corresponds to a statement node
        for &child_counter in child_counters {
            if let Some(&node_id) = self.node_ids.get(&child_counter) {
                // Check if this node is a statement using stored node_type
                if let Some(node_type) = self.node_types.get(&child_counter) {
                    if is_statement(node_type) {
                        return Some(node_id);
                    }
                }
            }
        }
        None
    }

    /// Get last statement node ID from child counters
    fn get_last_stmt_id_from_counters(&self, child_counters: &[usize]) -> Option<u32> {
        use crate::ast_parser::is_statement;
        
        // Find the last child counter that corresponds to a statement node
        for &child_counter in child_counters.iter().rev() {
            if let Some(&node_id) = self.node_ids.get(&child_counter) {
                if let Some(node_type) = self.node_types.get(&child_counter) {
                    if is_statement(node_type) {
                        return Some(node_id);
                    }
                }
            }
        }
        None
    }

}
