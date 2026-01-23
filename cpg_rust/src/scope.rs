//! Scope management for variable resolution

use indexmap::IndexMap;

/// Manages variable scopes for DFG construction
pub struct ScopeManager {
    scopes: Vec<IndexMap<String, u32>>, // name -> node_id
}

impl ScopeManager {
    pub fn new() -> Self {
        Self {
            scopes: vec![IndexMap::new()], // Global scope
        }
    }

    /// Enter a new scope (e.g., function body)
    pub fn enter_scope(&mut self) {
        self.scopes.push(IndexMap::new());
    }

    /// Exit current scope
    pub fn exit_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Define a symbol in the current scope
    pub fn define(&mut self, name: String, node_id: u32) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, node_id);
        }
    }

    /// Resolve a symbol (lookup from innermost to outermost scope)
    #[allow(dead_code)]  // Will be used in Phase 4 (DFG construction)
    pub fn resolve(&self, name: &str) -> Option<u32> {
        for scope in self.scopes.iter().rev() {
            if let Some(&node_id) = scope.get(name) {
                return Some(node_id);
            }
        }
        None
    }
}

impl Default for ScopeManager {
    fn default() -> Self {
        Self::new()
    }
}
