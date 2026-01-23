"""
OWASP Top 10 and CWE mapping module for NNAST.

This module provides mapping from internal pattern IDs to OWASP Top 10 categories
and CWE IDs, implementing the two-layer label structure described in the design document.
"""
import yaml
import pathlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class PatternMapping:
    """Mapping information for a vulnerability pattern."""
    pattern_id: str
    owasp: str
    cwe: List[str]
    description: str


class OWASPMapper:
    """
    Maps internal pattern IDs to OWASP Top 10 categories and CWE IDs.
    
    This implements the "two-layer label structure" from the design document:
    - Internal labels: Pattern IDs (for training)
    - External labels: OWASP Top 10 categories (for reporting)
    """
    
    def __init__(self, patterns_yaml_path: Optional[pathlib.Path] = None):
        """
        Initialize OWASP mapper.
        
        Args:
            patterns_yaml_path: Path to patterns.yaml. If None, looks for patterns.yaml
                               in the project root.
        """
        if patterns_yaml_path is None:
            # Try to find patterns.yaml in project root
            project_root = pathlib.Path(__file__).parent.parent
            patterns_yaml_path = project_root / "patterns.yaml"
        
        self.patterns_yaml_path = pathlib.Path(patterns_yaml_path)
        self.patterns: Dict[str, PatternMapping] = {}
        self._load_patterns()
    
    def _load_patterns(self):
        """Load pattern mappings from YAML file."""
        if not self.patterns_yaml_path.exists():
            raise FileNotFoundError(
                f"Patterns YAML not found: {self.patterns_yaml_path}"
            )
        
        with open(self.patterns_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Load pattern mappings
        if 'patterns' in data:
            for pattern_def in data['patterns']:
                pattern_id = pattern_def['id']
                owasp = pattern_def.get('owasp', 'Unknown')
                cwe_list = pattern_def.get('cwe', [])
                description = pattern_def.get('description', '')
                
                mapping = PatternMapping(
                    pattern_id=pattern_id,
                    owasp=owasp,
                    cwe=cwe_list if isinstance(cwe_list, list) else [cwe_list],
                    description=description
                )
                self.patterns[pattern_id] = mapping
    
    def get_owasp(self, pattern_id: str) -> Optional[str]:
        """
        Get OWASP Top 10 category for a pattern ID.
        
        Args:
            pattern_id: Internal pattern ID (e.g., "SQLI_RAW_STRING_FORMAT")
        
        Returns:
            OWASP category string (e.g., "A03: Injection") or None if not found
        """
        pattern = self.patterns.get(pattern_id)
        return pattern.owasp if pattern else None
    
    def get_cwe(self, pattern_id: str) -> List[str]:
        """
        Get CWE IDs for a pattern ID.
        
        Args:
            pattern_id: Internal pattern ID
        
        Returns:
            List of CWE IDs (e.g., ["CWE-89"]) or empty list if not found
        """
        pattern = self.patterns.get(pattern_id)
        return pattern.cwe if pattern else []
    
    def get_primary_cwe(self, pattern_id: str) -> Optional[str]:
        """
        Get primary CWE ID for a pattern ID (first CWE if multiple).
        
        Args:
            pattern_id: Internal pattern ID
        
        Returns:
            Primary CWE ID (e.g., "CWE-89") or None if not found
        """
        cwe_list = self.get_cwe(pattern_id)
        return cwe_list[0] if cwe_list else None
    
    def get_description(self, pattern_id: str) -> Optional[str]:
        """
        Get description for a pattern ID.
        
        Args:
            pattern_id: Internal pattern ID
        
        Returns:
            Description string or None if not found
        """
        pattern = self.patterns.get(pattern_id)
        return pattern.description if pattern else None
    
    def get_pattern_info(self, pattern_id: str) -> Optional[PatternMapping]:
        """
        Get complete pattern mapping information.
        
        Args:
            pattern_id: Internal pattern ID
        
        Returns:
            PatternMapping object or None if not found
        """
        return self.patterns.get(pattern_id)
    
    def format_result(
        self,
        pattern_id: str,
        confidence: float,
        file_path: str,
        lines: Optional[List[int]] = None,
        line_range: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format inference result according to design document 5.3 format.
        
        Args:
            pattern_id: Internal pattern ID
            confidence: Confidence score (0.0 to 1.0)
            file_path: File path where vulnerability was detected
            lines: List of line numbers (optional)
            line_range: Tuple of (start_line, end_line) (optional)
            **kwargs: Additional fields to include
        
        Returns:
            Formatted result dictionary matching design document 5.3 format:
            {
                "pattern_id": "SSRF_REQUESTS_URL_TAINTED",
                "cwe_id": "CWE-918",
                "owasp": "A10: SSRF",
                "confidence": 0.94,
                "location": {
                    "file": "views.py",
                    "lines": [42, 48]
                }
            }
        """
        pattern = self.patterns.get(pattern_id)
        
        if not pattern:
            # Fallback: use pattern_id as-is if not found
            owasp = "Unknown"
            cwe_id = None
            description = ""
        else:
            owasp = pattern.owasp
            cwe_id = pattern.cwe[0] if pattern.cwe else None
            description = pattern.description
        
        # Determine line information
        location_lines = None
        if lines:
            location_lines = lines
        elif line_range:
            start, end = line_range
            location_lines = list(range(start, end + 1))
        
        result = {
            "pattern_id": pattern_id,
            "owasp": owasp,
            "confidence": confidence,
            "location": {
                "file": file_path,
            }
        }
        
        if cwe_id:
            result["cwe_id"] = cwe_id
        
        if location_lines:
            result["location"]["lines"] = location_lines
        
        if description:
            result["description"] = description
        
        # Add any additional fields
        result.update(kwargs)
        
        return result
    
    def get_all_patterns(self) -> List[str]:
        """
        Get list of all registered pattern IDs.
        
        Returns:
            List of pattern IDs
        """
        return list(self.patterns.keys())
    
    def get_patterns_by_owasp(self, owasp: str) -> List[str]:
        """
        Get pattern IDs for a specific OWASP category.
        
        Args:
            owasp: OWASP category (e.g., "A03: Injection")
        
        Returns:
            List of pattern IDs matching the OWASP category
        """
        return [
            pattern_id
            for pattern_id, pattern in self.patterns.items()
            if pattern.owasp == owasp
        ]
    
    def get_patterns_by_cwe(self, cwe: str) -> List[str]:
        """
        Get pattern IDs for a specific CWE ID.
        
        Args:
            cwe: CWE ID (e.g., "CWE-89")
        
        Returns:
            List of pattern IDs matching the CWE ID
        """
        return [
            pattern_id
            for pattern_id, pattern in self.patterns.items()
            if cwe in pattern.cwe
        ]
    
    def infer_pattern_id(
        self,
        source_id: Optional[str] = None,
        sink_id: Optional[str] = None,
        sink_kind: Optional[str] = None,
        node_code: Optional[str] = None
    ) -> Optional[str]:
        """
        Infer pattern ID from source/sink information.
        
        This is a heuristic method that tries to match source/sink combinations
        to known patterns. It's not perfect but provides a reasonable guess.
        
        Args:
            source_id: Source ID from CPG node (e.g., "SRC_FLASK_REQUEST")
            sink_id: Sink ID from CPG node (e.g., "SINK_DBAPI_EXECUTE")
            sink_kind: Sink kind (e.g., "sql_exec", "cmd_exec", "http_client")
            node_code: Code string of the sink node (for additional heuristics)
        
        Returns:
            Most likely pattern ID or None if cannot be determined
        """
        if not sink_id and not sink_kind:
            return None
        
        # Load pattern definitions to check source/sink combinations
        import yaml
        with open(self.patterns_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if 'patterns' not in data:
            return None
        
        # Score patterns based on source/sink match
        scored_patterns = []
        
        for pattern_def in data['patterns']:
            pattern_id = pattern_def['id']
            pattern_sources = pattern_def.get('sources', [])
            pattern_sinks = pattern_def.get('sinks', [])
            
            score = 0
            
            # Check source match
            if source_id:
                if source_id in pattern_sources:
                    score += 10
                elif any(s.startswith(source_id.split('_')[0]) for s in pattern_sources):
                    score += 5
            
            # Check sink match
            if sink_id:
                for sink_def in pattern_sinks:
                    if sink_def.get('id') == sink_id:
                        score += 10
                        break
            
            # Check sink_kind match
            if sink_kind:
                for sink_def in pattern_sinks:
                    if sink_def.get('kind') == sink_kind:
                        score += 8
                        break
            
            # Additional heuristics based on code
            if node_code:
                code_lower = node_code.lower()
                pattern_id_lower = pattern_id.lower()
                
                # SQL injection patterns
                if 'sqli' in pattern_id_lower:
                    if any(keyword in code_lower for keyword in ['execute', 'query', 'cursor']):
                        score += 5
                
                # Command injection patterns
                if 'cmdi' in pattern_id_lower:
                    if any(keyword in code_lower for keyword in ['subprocess', 'system', 'popen']):
                        score += 5
                
                # SSRF patterns
                if 'ssrf' in pattern_id_lower:
                    if any(keyword in code_lower for keyword in ['requests', 'urlopen', 'httpx']):
                        score += 5
                
                # XSS patterns
                if 'xss' in pattern_id_lower:
                    if any(keyword in code_lower for keyword in ['markup', 'html', 'response']):
                        score += 5
            
            if score > 0:
                scored_patterns.append((pattern_id, score))
        
        # Return highest scoring pattern
        if scored_patterns:
            scored_patterns.sort(key=lambda x: x[1], reverse=True)
            return scored_patterns[0][0]
        
        return None

