#!/usr/bin/env python3
"""
Generate synthetic vulnerability code samples using LLM.

This script generates vulnerable and fixed code pairs for specific
vulnerability types that are difficult to collect from GitHub.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import tempfile

# Try to import OpenAI or Anthropic API
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class SyntheticVulnerability:
    """Synthetic vulnerability code pair."""
    vulnerability_type: str
    cwe_id: Optional[str]
    code_vulnerable: str
    code_fixed: str
    description: str
    metadata: Dict


class LLMCodeGenerator:
    """Generate code samples using LLM."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        """
        Initialize LLM code generator.
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name (e.g., "gpt-4", "claude-3-opus")
        """
        self.provider = provider
        self.model = model
        
        if provider == "openai" and not HAS_OPENAI:
            raise ImportError("openai package not installed. Install with: pip install openai")
        if provider == "anthropic" and not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    def generate_vulnerability_pair(
        self,
        vulnerability_type: str,
        context: Optional[str] = None
    ) -> Optional[Tuple[str, str, str]]:
        """
        Generate vulnerable and fixed code pair.
        
        Returns:
            Tuple of (vulnerable_code, fixed_code, description) or None
        """
        prompt = self._build_prompt(vulnerability_type, context)
        
        try:
            if self.provider == "openai":
                response = self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            return self._parse_response(response, vulnerability_type)
            
        except Exception as e:
            print(f"Error generating code: {e}")
            return None
    
    def _build_prompt(self, vulnerability_type: str, context: Optional[str]) -> str:
        """Build prompt for LLM."""
        base_prompt = f"""Generate a Python code example demonstrating a {vulnerability_type} vulnerability.

Requirements:
1. Write a realistic Python function or code snippet that contains a {vulnerability_type} vulnerability
2. The code should be practical and similar to real-world vulnerable code
3. Include a fixed version that properly mitigates the vulnerability
4. Provide a brief description of the vulnerability

Format your response as JSON:
{{
    "vulnerable_code": "the vulnerable Python code",
    "fixed_code": "the fixed Python code",
    "description": "brief description of the vulnerability and fix"
}}

The code should be complete and runnable (if it's a function, include necessary imports).
"""
        
        if context:
            base_prompt += f"\n\nContext: {context}"
        
        return base_prompt
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a security expert who creates realistic vulnerability examples for educational purposes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        message = client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.7,
            system="You are a security expert who creates realistic vulnerability examples for educational purposes.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    
    def _parse_response(self, response: str, vulnerability_type: str) -> Optional[Tuple[str, str, str]]:
        """Parse LLM response."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return (
                    data.get("vulnerable_code", ""),
                    data.get("fixed_code", ""),
                    data.get("description", f"{vulnerability_type} vulnerability")
                )
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response: {response[:200]}...")
        
        return None


class SyntheticDatasetGenerator:
    """Generate synthetic vulnerability dataset."""
    
    def __init__(self, output_dir: Path, llm_generator: LLMCodeGenerator):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm_generator = llm_generator
        
        # Output paths
        self.code_dir = self.output_dir / "code"
        self.processed_dir = self.output_dir / "processed"
        self.metadata_file = self.output_dir / "metadata.jsonl"
        
        self.code_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
    
    def generate_for_type(
        self,
        vulnerability_type: str,
        count: int = 10,
        cwe_id: Optional[str] = None
    ) -> List[SyntheticVulnerability]:
        """Generate synthetic samples for a vulnerability type."""
        print(f"Generating {count} {vulnerability_type} samples...")
        
        samples = []
        for i in range(count):
            print(f"  [{i+1}/{count}] Generating sample...")
            
            result = self.llm_generator.generate_vulnerability_pair(vulnerability_type)
            if not result:
                print(f"    Warning: Failed to generate sample")
                continue
            
            vulnerable_code, fixed_code, description = result
            
            sample = SyntheticVulnerability(
                vulnerability_type=vulnerability_type,
                cwe_id=cwe_id,
                code_vulnerable=vulnerable_code,
                code_fixed=fixed_code,
                description=description,
                metadata={
                    "synthetic": True,
                    "generated_by": self.llm_generator.provider,
                    "model": self.llm_generator.model,
                }
            )
            
            samples.append(sample)
            self._save_sample(sample, i)
            
            # Rate limiting
            import time
            time.sleep(1)
        
        print(f"Generated {len(samples)} samples")
        return samples
    
    def _save_sample(self, sample: SyntheticVulnerability, index: int):
        """Save a synthetic sample."""
        # Save code files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        vulnerable_file = self.code_dir / f"synthetic_{sample.vulnerability_type}_{index}_{timestamp}_vulnerable.py"
        fixed_file = self.code_dir / f"synthetic_{sample.vulnerability_type}_{index}_{timestamp}_fixed.py"
        
        vulnerable_file.write_text(sample.code_vulnerable, encoding="utf-8")
        fixed_file.write_text(sample.code_fixed, encoding="utf-8")
        
        # Generate CPG graphs
        graph_vulnerable = self._generate_cpg(vulnerable_file)
        graph_fixed = self._generate_cpg(fixed_file)
        
        # Save CPG graphs
        if graph_vulnerable:
            graph_file_vuln = self.processed_dir / f"synthetic_{sample.vulnerability_type}_{index}_{timestamp}_vulnerable.jsonl"
            with open(graph_file_vuln, "w") as f:
                f.write(json.dumps(graph_vulnerable, ensure_ascii=False) + "\n")
        
        if graph_fixed:
            graph_file_fixed = self.processed_dir / f"synthetic_{sample.vulnerability_type}_{index}_{timestamp}_fixed.jsonl"
            with open(graph_file_fixed, "w") as f:
                f.write(json.dumps(graph_fixed, ensure_ascii=False) + "\n")
        
        # Save metadata
        metadata = {
            "vulnerability_type": sample.vulnerability_type,
            "cwe_id": sample.cwe_id,
            "description": sample.description,
            "code_vulnerable_file": str(vulnerable_file.relative_to(self.output_dir)),
            "code_fixed_file": str(fixed_file.relative_to(self.output_dir)),
            "synthetic": True,
            "generated_by": self.llm_generator.provider,
            "model": self.llm_generator.model,
            "timestamp": datetime.now().isoformat(),
        }
        
        with open(self.metadata_file, "a") as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    
    def _generate_cpg(self, code_file: Path) -> Optional[Dict]:
        """Generate CPG graph from code file."""
        import subprocess
        import sys
        import tempfile
        
        try:
            # Use temporary file for output since /dev/stdout might not work on all systems
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_out:
                tmp_out_path = tmp_out.name
            
            result = subprocess.run(
                [sys.executable, "-m", "cli", str(code_file), "--out", tmp_out_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                with open(tmp_out_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        graph_data = json.loads(lines[0])
                        # Cleanup
                        Path(tmp_out_path).unlink(missing_ok=True)
                        return graph_data
            else:
                print(f"  Warning: CPG generation failed: {result.stderr}")
                Path(tmp_out_path).unlink(missing_ok=True)
        except Exception as e:
            print(f"  Warning: Failed to generate CPG: {e}")
            # Cleanup on error
            try:
                Path(tmp_out_path).unlink(missing_ok=True)
            except:
                pass
        
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic vulnerability code samples using LLM"
    )
    parser.add_argument(
        "--type",
        required=True,
        help="Vulnerability type (e.g., 'XSS', 'SSRF', 'Deserialization')"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset_synthetic",
        help="Output directory"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider"
    )
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Model name (e.g., 'gpt-4', 'claude-3-opus')"
    )
    parser.add_argument(
        "--cwe-id",
        help="CWE ID (e.g., 'CWE-79' for XSS)"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if args.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: OPENAI_API_KEY not set")
            print("Set it with: export OPENAI_API_KEY=your_key")
            return 1
    elif args.provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY not set")
            print("Set it with: export ANTHROPIC_API_KEY=your_key")
            return 1
    
    # Initialize generator
    llm_generator = LLMCodeGenerator(provider=args.provider, model=args.model)
    dataset_generator = SyntheticDatasetGenerator(Path(args.output_dir), llm_generator)
    
    # Generate samples
    samples = dataset_generator.generate_for_type(
        args.type,
        args.count,
        args.cwe_id
    )
    
    print(f"\nGenerated {len(samples)} synthetic samples")
    print(f"Output directory: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

