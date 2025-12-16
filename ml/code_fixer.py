#!/usr/bin/env python3
"""
LLM-based code fixer for generating vulnerability fix suggestions.

This module uses LLM to generate fix suggestions for vulnerable code
detected by the GNN model.
"""
import os
import json
import re
import pathlib
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

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


def load_env_file(env_path: Optional[pathlib.Path] = None) -> None:
    """
    Load environment variables from .env file.
    
    Simple implementation without external dependencies.
    """
    if env_path is None:
        # Look for .env in project root
        project_root = pathlib.Path(__file__).parent.parent
        env_path = project_root / ".env"
    
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE format
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Only set if not already in environment
                    if key and key not in os.environ:
                        os.environ[key] = value


@dataclass
class FixSuggestion:
    """Code fix suggestion from LLM."""
    fixed_code: str
    explanation: str
    vulnerability_type: Optional[str] = None
    confidence: Optional[float] = None


class LLMCodeFixer:
    """Generate code fix suggestions using LLM."""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        """
        Initialize LLM code fixer.
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name (e.g., "gpt-4o", "claude-3-opus")
        """
        # Load environment variables from .env file
        load_env_file()
        
        self.provider = provider
        self.model = model
        
        if provider == "openai" and not HAS_OPENAI:
            raise ImportError("openai package not installed. Install with: pip install openai")
        if provider == "anthropic" and not HAS_ANTHROPIC:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    def generate_fix(
        self,
        vulnerable_code: str,
        file_path: Optional[str] = None,
        vulnerability_type: Optional[str] = None,
        context: Optional[str] = None
    ) -> Optional[FixSuggestion]:
        """
        Generate fix suggestion for vulnerable code.
        
        Args:
            vulnerable_code: The vulnerable code to fix
            file_path: Path to the file (for context)
            vulnerability_type: Type of vulnerability (if known)
            context: Additional context about the vulnerability
            
        Returns:
            FixSuggestion object, or None on error
        """
        prompt = self._build_prompt(vulnerable_code, file_path, vulnerability_type, context)
        
        try:
            if self.provider == "openai":
                response = self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            return self._parse_response(response, vulnerability_type)
            
        except Exception as e:
            print(f"Error generating fix: {e}")
            return None
    
    def _build_prompt(
        self,
        vulnerable_code: str,
        file_path: Optional[str],
        vulnerability_type: Optional[str],
        context: Optional[str]
    ) -> str:
        """Build prompt for LLM."""
        base_prompt = """You are a security expert reviewing Python code for vulnerabilities.

Your task is to:
1. Identify the security vulnerability in the provided code
2. Generate a secure, fixed version of the code
3. Provide a clear explanation of the vulnerability and how the fix addresses it

"""
        
        if vulnerability_type:
            base_prompt += f"**Vulnerability Type**: {vulnerability_type}\n\n"
        
        if file_path:
            base_prompt += f"**File Path**: {file_path}\n\n"
        
        if context:
            base_prompt += f"**Additional Context**: {context}\n\n"
        
        base_prompt += f"""**Vulnerable Code**:
```python
{vulnerable_code}
```

Please provide:
1. A fixed version of the code that addresses the security vulnerability
2. A clear explanation of:
   - What the vulnerability is
   - Why it's dangerous
   - How the fix mitigates the vulnerability

Format your response as JSON:
{{
    "fixed_code": "the fixed Python code (complete, runnable code)",
    "explanation": "detailed explanation of the vulnerability and fix",
    "vulnerability_type": "type of vulnerability (e.g., SQL Injection, XSS, Command Injection)",
    "confidence": 0.95
}}

The fixed code should:
- Be complete and runnable (include necessary imports)
- Maintain the same functionality as the original code
- Follow Python best practices
- Be secure against the identified vulnerability
"""
        
        return base_prompt
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a security expert specializing in Python code security. You provide clear, actionable fixes for security vulnerabilities."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent, focused responses
            max_tokens=3000
        )
        
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        message = client.messages.create(
            model=self.model,
            max_tokens=3000,
            temperature=0.3,
            system="You are a security expert specializing in Python code security. You provide clear, actionable fixes for security vulnerabilities.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    
    def _parse_response(
        self,
        response: str,
        vulnerability_type: Optional[str]
    ) -> Optional[FixSuggestion]:
        """Parse LLM response."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return FixSuggestion(
                    fixed_code=data.get("fixed_code", ""),
                    explanation=data.get("explanation", ""),
                    vulnerability_type=data.get("vulnerability_type") or vulnerability_type,
                    confidence=data.get("confidence")
                )
            
            # Fallback: try to extract code block and explanation separately
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                fixed_code = code_match.group(1).strip()
                # Use the rest as explanation
                explanation = response.replace(f"```python\n{fixed_code}\n```", "").strip()
                return FixSuggestion(
                    fixed_code=fixed_code,
                    explanation=explanation or "Fix generated by LLM",
                    vulnerability_type=vulnerability_type,
                    confidence=0.8
                )
            
            print(f"Warning: Could not parse LLM response. Response: {response[:200]}...")
            return None
            
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response: {response[:200]}...")
            return None

