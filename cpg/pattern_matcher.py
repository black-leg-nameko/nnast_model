"""
Pattern matcher for CPG nodes based on patterns.yaml definitions.

This module loads YAML pattern definitions and matches CPG nodes against
source/sink/sanitizer signatures.
"""
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
    
    def match_source(self, node_code: str, node_kind: str, frameworks: Optional[Set[str]] = None) -> Optional[str]:
        """
        Match a node against source patterns.
        
        Args:
            node_code: Code string of the node
            node_kind: Node kind (e.g., "Call", "Attribute")
            frameworks: Set of detected frameworks (e.g., {"flask", "django"})
        
        Returns:
            Source ID if matched, None otherwise
        """
        frameworks = frameworks or set()
        
        for source_id, source in self.sources.items():
            # Check framework compatibility
            if source.frameworks:
                if not any(fw in frameworks for fw in source.frameworks):
                    continue
            
            # Match calls
            if source.match_calls and node_kind == "Call":
                if self._match_call_pattern(node_code, source.match_calls):
                    return source_id
            
            # Match attributes
            if source.match_attrs:
                if node_kind == "Attribute":
                    if self._match_attr_pattern(node_code, source.match_attrs):
                        return source_id
                elif node_kind == "Call":
                    # For method calls, also check if the base attribute matches
                    # e.g., "request.args.get('id')" -> check "request.args"
                    base_attr = self._extract_base_attr_from_call(node_code)
                    if base_attr and self._match_attr_pattern(base_attr, source.match_attrs):
                        return source_id
        
        return None
    
    def match_sink(self, node_code: str, node_kind: str, frameworks: Optional[Set[str]] = None) -> Optional[Tuple[str, str]]:
        """
        Match a node against sink patterns.
        
        Args:
            node_code: Code string of the node
            node_kind: Node kind (e.g., "Call")
            frameworks: Set of detected frameworks
        
        Returns:
            Tuple of (sink_id, sink_kind) if matched, None otherwise
        """
        frameworks = frameworks or set()
        
        for sink_id, sink in self.sinks.items():
            # Check framework compatibility
            if sink.frameworks:
                if not any(fw in frameworks for fw in sink.frameworks):
                    continue
            
            # Match calls
            if sink.match_calls and node_kind == "Call":
                if self._match_call_pattern(node_code, sink.match_calls):
                    # TODO: Check constraints (e.g., shell=True, verify=False)
                    # For now, we skip constraint checking
                    return (sink_id, sink.kind)
            
            # Match attributes
            if sink.match_attrs and node_kind == "Attribute":
                if self._match_attr_pattern(node_code, sink.match_attrs):
                    return (sink_id, sink.kind)
        
        return None
    
    def match_sanitizer(self, node_code: str, node_kind: str) -> Optional[Tuple[str, str]]:
        """
        Match a node against sanitizer patterns.
        
        Args:
            node_code: Code string of the node
            node_kind: Node kind (e.g., "Call")
        
        Returns:
            Tuple of (sanitizer_id, sanitizer_kind) if matched, None otherwise
        """
        for sanitizer_id, sanitizer in self.sanitizers.items():
            # Match calls
            if sanitizer.match_calls and node_kind == "Call":
                if self._match_call_pattern(node_code, sanitizer.match_calls):
                    return (sanitizer_id, sanitizer.kind)
            
            # Match attributes
            if sanitizer.match_attrs and node_kind == "Attribute":
                if self._match_attr_pattern(node_code, sanitizer.match_attrs):
                    return (sanitizer_id, sanitizer.kind)
        
        return None
    
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

