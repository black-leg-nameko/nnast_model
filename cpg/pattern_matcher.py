"""
Pattern matcher for CPG nodes based on patterns.yaml definitions.

This module loads YAML pattern definitions and matches CPG nodes against
source/sink/sanitizer signatures.
"""
import ast
import yaml
import pathlib
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass


@dataclass
class SourceDefinition:
    """Source definition from patterns.yaml"""
    id: str
    kinds: List[str]
    frameworks: Optional[List[str]] = None
    match_calls: Optional[List[str]] = None
    match_attrs: Optional[List[str]] = None


@dataclass
class SinkDefinition:
    """Sink definition from patterns.yaml"""
    id: str
    kind: str
    frameworks: Optional[List[str]] = None
    match_calls: Optional[List[str]] = None
    match_attrs: Optional[List[str]] = None
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class SanitizerDefinition:
    """Sanitizer definition from patterns.yaml"""
    id: str
    kind: str
    match_calls: Optional[List[str]] = None
    match_attrs: Optional[List[str]] = None


class PatternMatcher:
    """
    Matches CPG nodes against source/sink/sanitizer patterns from YAML.
    """
    
    def __init__(self, patterns_yaml_path: Optional[pathlib.Path] = None):
        """
        Initialize pattern matcher.
        
        Args:
            patterns_yaml_path: Path to patterns.yaml. If None, looks for patterns.yaml
                               in the project root.
        """
        if patterns_yaml_path is None:
            # Try to find patterns.yaml in project root
            project_root = pathlib.Path(__file__).parent.parent
            patterns_yaml_path = project_root / "patterns.yaml"
        
        self.patterns_yaml_path = pathlib.Path(patterns_yaml_path)
        self.sources: Dict[str, SourceDefinition] = {}
        self.sinks: Dict[str, SinkDefinition] = {}
        self.sanitizers: Dict[str, SanitizerDefinition] = {}
        self._load_patterns()
    
    def _load_patterns(self):
        """Load patterns from YAML file."""
        if not self.patterns_yaml_path.exists():
            raise FileNotFoundError(
                f"Patterns YAML not found: {self.patterns_yaml_path}"
            )
        
        with open(self.patterns_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Load common sources
        if 'common' in data and 'sources' in data['common']:
            for src_def in data['common']['sources']:
                source = SourceDefinition(
                    id=src_def['id'],
                    kinds=src_def.get('kinds', []),
                    frameworks=src_def.get('frameworks'),
                    match_calls=src_def.get('match', {}).get('calls'),
                    match_attrs=src_def.get('match', {}).get('attrs'),
                )
                self.sources[source.id] = source
        
        # Load common sanitizers
        if 'common' in data and 'sanitizers' in data['common']:
            for san_def in data['common']['sanitizers']:
                sanitizer = SanitizerDefinition(
                    id=san_def['id'],
                    kind=san_def.get('kind', ''),
                    match_calls=san_def.get('match', {}).get('calls'),
                    match_attrs=san_def.get('match', {}).get('attrs'),
                )
                self.sanitizers[sanitizer.id] = sanitizer
        
        # Load pattern-specific sinks
        if 'patterns' in data:
            for pattern in data['patterns']:
                if 'sinks' in pattern:
                    for sink_def in pattern['sinks']:
                        sink = SinkDefinition(
                            id=sink_def['id'],
                            kind=sink_def.get('kind', ''),
                            frameworks=pattern.get('frameworks'),
                            match_calls=sink_def.get('match', {}).get('calls'),
                            match_attrs=sink_def.get('match', {}).get('attrs'),
                            constraints=sink_def.get('constraints'),
                        )
                        self.sinks[sink.id] = sink
    
    def _normalize_call_name(self, call_code: str) -> str:
        """
        Normalize call code to match pattern.
        
        Examples:
            "os.getenv('KEY')" -> "os.getenv"
            "flask.request.args.get('id')" -> "flask.request.args.get"
            "cursor.execute(query)" -> "cursor.execute"
            "request.args.get('id')" -> "request.args.get"
        """
        # Remove arguments and whitespace
        call_code = call_code.strip()
        
        # Extract function/method name part (before first '(')
        if '(' in call_code:
            call_code = call_code[:call_code.index('(')]
        
        # Remove trailing dots if any
        call_code = call_code.rstrip('.')
        
        return call_code
    
    def _extract_base_attr_from_call(self, call_code: str) -> Optional[str]:
        """
        Extract base attribute from method call for attribute-based source matching.
        
        Examples:
            "request.args.get('id')" -> "request.args"
            "flask.request.args.get('id')" -> "flask.request.args"
        """
        call_code = call_code.strip()
        if '(' in call_code:
            call_code = call_code[:call_code.index('(')]
        
        # If it's a method call (has at least one dot), try to extract the base
        if '.' in call_code:
            # Split by dots and remove the last part (method name)
            parts = call_code.split('.')
            if len(parts) >= 2:
                # Return everything except the last part
                return '.'.join(parts[:-1])
        
        return None
    
    def _normalize_attr_name(self, attr_code: str) -> str:
        """
        Normalize attribute access code to match pattern.
        
        Examples:
            "flask.request.args" -> "flask.request.args"
            "request.GET" -> "request.GET"
        """
        # Remove method calls (e.g., ".get()")
        if '.' in attr_code and '(' in attr_code:
            # Find the last dot before the first parenthesis
            paren_idx = attr_code.index('(')
            last_dot_before_paren = attr_code.rfind('.', 0, paren_idx)
            if last_dot_before_paren != -1:
                attr_code = attr_code[:last_dot_before_paren]
        
        return attr_code.strip()
    
    def _match_call_pattern(self, call_code: str, patterns: List[str]) -> bool:
        """Check if call code matches any pattern."""
        normalized = self._normalize_call_name(call_code)
        for pattern in patterns:
            # Exact match
            if normalized == pattern:
                return True
            # Match if normalized ends with the full pattern (e.g., "module.func" matches "module.func")
            if normalized.endswith('.' + pattern) or normalized == pattern.split('.')[-1]:
                # But only if it's a proper match (not partial)
                # e.g., "os.getenv" matches "os.getenv", but "request.args.get" doesn't match "os.getenv"
                pattern_parts = pattern.split('.')
                normalized_parts = normalized.split('.')
                if len(normalized_parts) >= len(pattern_parts):
                    # Check if the last N parts match
                    if normalized_parts[-len(pattern_parts):] == pattern_parts:
                        return True
        return False
    
    def _match_attr_pattern(self, attr_code: str, patterns: List[str]) -> bool:
        """Check if attribute code matches any pattern."""
        normalized = self._normalize_attr_name(attr_code)
        for pattern in patterns:
            if normalized == pattern:
                return True
        return False
    
    def match_source(self, node_code: str, node_kind: str, frameworks: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Match a node against source patterns.
        
        Args:
            node_code: Code string of the node
            node_kind: Node kind (e.g., "Call", "Attribute")
            frameworks: Set of detected frameworks (e.g., {"flask", "django"})
        
        Returns:
            Dict with source_id and source_kinds if matched, None otherwise
            Format: {"source_id": str, "source_kinds": List[str]}
        """
        frameworks = frameworks or set()
        
        for source_id, source in self.sources.items():
            # Check framework compatibility
            if source.frameworks:
                if not any(fw in frameworks for fw in source.frameworks):
                    continue
            
            matched = False
            
            # Match calls
            if source.match_calls and node_kind == "Call":
                if self._match_call_pattern(node_code, source.match_calls):
                    matched = True
            
            # Match attributes
            if source.match_attrs and not matched:
                if node_kind == "Attribute":
                    if self._match_attr_pattern(node_code, source.match_attrs):
                        matched = True
                elif node_kind == "Call":
                    # For method calls, also check if the base attribute matches
                    # e.g., "request.args.get('id')" -> check "request.args"
                    base_attr = self._extract_base_attr_from_call(node_code)
                    if base_attr and self._match_attr_pattern(base_attr, source.match_attrs):
                        matched = True
            
            if matched:
                return {
                    "source_id": source_id,
                    "source_kinds": source.kinds
                }
        
        return None
    
    def match_sink(
        self, 
        node_code: str, 
        node_kind: str, 
        frameworks: Optional[Set[str]] = None,
        ast_node: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Match a node against sink patterns.
        
        Args:
            node_code: Code string of the node
            node_kind: Node kind (e.g., "Call")
            frameworks: Set of detected frameworks
            ast_node: Optional AST node for constraint checking
        
        Returns:
            Dict with sink_id, sink_kind, and constraint_satisfied if matched, None otherwise
            Format: {"sink_id": str, "sink_kind": str, "constraint_satisfied": bool}
        """
        frameworks = frameworks or set()
        
        for sink_id, sink in self.sinks.items():
            # Check framework compatibility
            if sink.frameworks:
                if not any(fw in frameworks for fw in sink.frameworks):
                    continue
            
            matched = False
            
            # Match calls
            if sink.match_calls and node_kind == "Call":
                if self._match_call_pattern(node_code, sink.match_calls):
                    matched = True
            
            # Match attributes
            if sink.match_attrs and node_kind == "Attribute":
                if self._match_attr_pattern(node_code, sink.match_attrs):
                    matched = True
            
            if matched:
                # Check constraints if sink has them
                constraint_satisfied = True
                if sink.constraints and ast_node is not None:
                    constraint_satisfied = self._check_constraints(ast_node, sink.constraints)
                
                return {
                    "sink_id": sink_id,
                    "sink_kind": sink.kind,
                    "constraint_satisfied": constraint_satisfied
                }
        
        return None
    
    def _check_constraints(self, ast_node: Any, constraints: Dict[str, Any]) -> bool:
        """
        Check if AST node satisfies sink constraints.
        
        Args:
            ast_node: AST node (should be ast.Call for function calls)
            constraints: Constraints dict from YAML
        
        Returns:
            True if constraints are satisfied, False otherwise
        """
        if not isinstance(ast_node, ast.Call):
            return False
        
        # Check kwargs constraints (e.g., shell=True, verify=False)
        if "kwargs" in constraints:
            required_kwargs = constraints["kwargs"]
            # Build dict of actual keyword arguments
            actual_kwargs = {}
            for kw in ast_node.keywords:
                if isinstance(kw.arg, str):
                    # Get the value (simplified: check for True/False constants)
                    if isinstance(kw.value, ast.Constant):
                        actual_kwargs[kw.arg] = kw.value.value
                    # Note: ast.NameConstant was removed in Python 3.8, using ast.Constant only
            
            # Check if all required kwargs match
            for key, required_value in required_kwargs.items():
                if key not in actual_kwargs:
                    return False
                if actual_kwargs[key] != required_value:
                    return False
        
        # Check any_of constraints (e.g., multiple constraint options)
        if "any_of" in constraints:
            for constraint_option in constraints["any_of"]:
                if self._check_constraints(ast_node, constraint_option):
                    return True
            return False
        
        return True
    
    def match_sanitizer(
        self, 
        node_code: str, 
        node_kind: str,
        ast_node: Optional[Any] = None
    ) -> Optional[Dict[str, str]]:
        """
        Match a node against sanitizer patterns.
        
        Args:
            node_code: Code string of the node
            node_kind: Node kind (e.g., "Call")
            ast_node: Optional AST node for conditional sanitizer checks
        
        Returns:
            Dict with sanitizer_id and sanitizer_kind if matched, None otherwise
            Format: {"sanitizer_id": str, "sanitizer_kind": str}
        """
        for sanitizer_id, sanitizer in self.sanitizers.items():
            # Match calls
            if sanitizer.match_calls and node_kind == "Call":
                if self._match_call_pattern(node_code, sanitizer.match_calls):
                    # Special handling for conditional sanitizers
                    if sanitizer_id == "SAN_SQL_PARAM":
                        # Check if this is actually a parameterized query
                        if not self._check_sql_parameterization(ast_node):
                            continue  # Skip if not parameterized
                    
                    return {
                        "sanitizer_id": sanitizer_id,
                        "sanitizer_kind": sanitizer.kind
                    }
            
            # Match attributes
            if sanitizer.match_attrs and node_kind == "Attribute":
                if self._match_attr_pattern(node_code, sanitizer.match_attrs):
                    return {
                        "sanitizer_id": sanitizer_id,
                        "sanitizer_kind": sanitizer.kind
                    }
        
        return None
    
    def _check_sql_parameterization(self, ast_node: Optional[Any]) -> bool:
        """
        Check if SQL execute call uses parameterized queries.
        
        A parameterized query must:
        1. Have a query string with placeholders (%s, ?, :name)
        2. Have user input passed as params argument (not in query string)
        
        Args:
            ast_node: AST Call node for cursor.execute(query, params)
        
        Returns:
            True if this is a parameterized query, False otherwise
        """
        if not isinstance(ast_node, ast.Call):
            return False
        
        # cursor.execute(query, params) typically has 2+ arguments
        if len(ast_node.args) < 2:
            return False
        
        # First argument should be the query string
        query_arg = ast_node.args[0]
        
        # Try to extract query string (simplified: only handles string literals)
        query_string = None
        if isinstance(query_arg, ast.Constant) and isinstance(query_arg.value, str):
            query_string = query_arg.value
        elif isinstance(query_arg, ast.JoinedStr):
            # f-string: not parameterized (user input in query string)
            return False
        
        # If we can't extract the query string statically, be conservative
        if query_string is None:
            # Check if second argument exists (params)
            # If params exist, assume it might be parameterized
            return len(ast_node.args) >= 2
        
        # Check for placeholders in query string
        has_placeholder = False
        placeholder_patterns = ['%s', '%d', '%f', '?', ':name', ':id', ':value']
        
        for pattern in placeholder_patterns:
            if pattern in query_string:
                has_placeholder = True
                break
        
        # Also check for named placeholders (e.g., %(name)s)
        import re
        if re.search(r'%\([^)]+\)[sd]', query_string):
            has_placeholder = True
        
        # Must have placeholder AND params argument
        return has_placeholder and len(ast_node.args) >= 2
    
    def detect_frameworks(self, source_code: str) -> Set[str]:
        """
        Detect frameworks from source code (simple heuristic).
        
        Args:
            source_code: Full source code of the file
        
        Returns:
            Set of detected framework names
        """
        frameworks = set()
        source_lower = source_code.lower()
        
        # Flask detection
        if any(pattern in source_lower for pattern in [
            'from flask import', 'import flask', 'flask.', '@app.route'
        ]):
            frameworks.add('flask')
        
        # Django detection
        if any(pattern in source_lower for pattern in [
            'from django', 'import django', 'django.', 'from django.http'
        ]):
            frameworks.add('django')
        
        # FastAPI detection
        if any(pattern in source_lower for pattern in [
            'from fastapi import', 'import fastapi', 'fastapi.', '@app.get', '@app.post'
        ]):
            frameworks.add('fastapi')
        
        # Starlette detection (often used with FastAPI)
        if any(pattern in source_lower for pattern in [
            'from starlette', 'import starlette', 'starlette.'
        ]):
            frameworks.add('starlette')
        
        return frameworks

