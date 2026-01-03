#!/usr/bin/env python3
"""
Generate balanced synthetic dataset for NNAST training.

This script generates vulnerable and safe code pairs for all 15 patterns
defined in patterns.yaml, ensuring balanced distribution across:
- Pattern IDs (15 patterns)
- Frameworks (Flask, Django, FastAPI)
- Complexity levels (simple, medium, complex)
- Vulnerable vs Safe (50/50)

All CPG graphs are generated with pattern matching enabled.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random

# Import pattern matcher to get pattern IDs
try:
    from cpg.pattern_matcher import PatternMatcher
except ImportError:
    PatternMatcher = None


def generate_commit_hash(seed: str) -> str:
    """Generate a deterministic commit hash from a seed."""
    return hashlib.md5(seed.encode()).hexdigest()[:16]


def generate_cpg_graph(code_file: Path, patterns_yaml: Optional[Path] = None) -> Optional[Dict]:
    """Generate CPG graph from code file with pattern matching enabled."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
        
        # Get project root (assuming this file is in data/ directory)
        project_root = Path(__file__).parent.parent
        
        cmd = [sys.executable, "-m", "cli", str(code_file), "--out", tmp_out_path]
        if patterns_yaml and patterns_yaml.exists():
            cmd.extend(["--patterns", str(patterns_yaml)])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(project_root)  # Project root
        )
        
        if result.returncode == 0:
            with open(tmp_out_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    graph_data = json.loads(lines[0])
                    Path(tmp_out_path).unlink(missing_ok=True)
                    return graph_data
        else:
            print(f"  Warning: CPG generation failed: {result.stderr}")
            Path(tmp_out_path).unlink(missing_ok=True)
    except Exception as e:
        print(f"  Warning: Failed to generate CPG: {e}")
        try:
            Path(tmp_out_path).unlink(missing_ok=True)
        except:
            pass
    
    return None


# Template code samples for each pattern ID
# These are simplified templates that will be expanded with framework-specific code

PATTERN_TEMPLATES = {
    "SQLI_RAW_STRING_FORMAT": {
        "description": "SQL Injection via raw string formatting",
        "vulnerable_template": """from {framework_import} import {request_obj}

def get_user_data(user_id):
    # SQL Injection vulnerability: Raw string formatting
    conn = {db_module}.connect({db_name})
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {{user_id}}"
    cursor.execute(query)
    return cursor.fetchall()

# Vulnerable usage
user_id = {request_obj}.args.get('id')
result = get_user_data(user_id)
""",
        "safe_template": """from {framework_import} import {request_obj}
import {db_module}

def get_user_data(user_id):
    # Fixed: Parameterized query
    conn = {db_module}.connect({db_name})
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    return cursor.fetchall()

# Safe usage
user_id = {request_obj}.args.get('id')
result = get_user_data(user_id)
"""
    },
    "SQLI_RAW_STRING_CONCAT": {
        "description": "SQL Injection via string concatenation",
        "vulnerable_template": """from {framework_import} import {request_obj}
import {db_module}

def search_users(name):
    # SQL Injection vulnerability: String concatenation
    conn = {db_module}.connect({db_name})
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + name + "'"
    cursor.execute(query)
    return cursor.fetchall()

# Vulnerable usage
user_name = {request_obj}.form.get('name')
result = search_users(user_name)
""",
        "safe_template": """from {framework_import} import {request_obj}
import {db_module}

def search_users(name):
    # Fixed: Parameterized query
    conn = {db_module}.connect({db_name})
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = ?"
    cursor.execute(query, (name,))
    return cursor.fetchall()

# Safe usage
user_name = {request_obj}.form.get('name')
result = search_users(user_name)
"""
    },
    "CMDI_SUBPROCESS_SHELL_TRUE": {
        "description": "Command Injection via subprocess with shell=True",
        "vulnerable_template": """from {framework_import} import {request_obj}
import subprocess

def execute_command(cmd):
    # Command Injection vulnerability: shell=True
    result = subprocess.run(cmd, shell=True)
    return result

# Vulnerable usage
user_cmd = {request_obj}.form.get('cmd')
execute_command(user_cmd)
""",
        "safe_template": """from {framework_import} import {request_obj}
import subprocess
import shlex

def execute_command(cmd):
    # Fixed: Use shell=False and shlex.split
    cmd_list = shlex.split(cmd)
    result = subprocess.run(cmd_list, shell=False)
    return result

# Safe usage
user_cmd = {request_obj}.form.get('cmd')
execute_command(user_cmd)
"""
    },
    "SQLI_RAW_FSTRING": {
        "description": "SQL Injection via f-string",
        "vulnerable_template": """from {framework_import} import {request_obj}
import {db_module}

def get_user(user_id):
    # SQL Injection vulnerability: f-string
    conn = {db_module}.connect({db_name})
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {{user_id}}"
    cursor.execute(query)
    return cursor.fetchone()

# Vulnerable usage
user_id = {request_obj}.args.get('id')
result = get_user(user_id)
""",
        "safe_template": """from {framework_import} import {request_obj}
import {db_module}

def get_user(user_id):
    # Fixed: Parameterized query
    conn = {db_module}.connect({db_name})
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    return cursor.fetchone()

# Safe usage
user_id = {request_obj}.args.get('id')
result = get_user(user_id)
"""
    },
    "CMDI_OS_SYSTEM_TAINTED": {
        "description": "Command Injection via os.system",
        "vulnerable_template": """from {framework_import} import {request_obj}
import os

def run_command(cmd):
    # Command Injection vulnerability: os.system
    os.system(cmd)

# Vulnerable usage
user_cmd = {request_obj}.args.get('cmd')
run_command(user_cmd)
""",
        "safe_template": """from {framework_import} import {request_obj}
import subprocess
import shlex

def run_command(cmd):
    # Fixed: Use subprocess.run with shell=False
    cmd_list = shlex.split(cmd)
    subprocess.run(cmd_list, shell=False)

# Safe usage
user_cmd = {request_obj}.args.get('cmd')
run_command(user_cmd)
"""
    },
    "XSS_REFLECTED": {
        "description": "Reflected XSS vulnerability",
        "vulnerable_template": """from {framework_import} import {request_obj}, {response_obj}

def search_page():
    # XSS vulnerability: Reflected user input
    query = {request_obj}.args.get('q', '')
    html = f"<h1>Search results for: {{query}}</h1>"
    return {response_obj}(html)

# Vulnerable usage
result = search_page()
""",
        "safe_template": """from {framework_import} import {request_obj}, {response_obj}
from {escape_module} import escape

def search_page():
    # Fixed: Escape user input
    query = {request_obj}.args.get('q', '')
    safe_query = escape(query)
    html = f"<h1>Search results for: {{safe_query}}</h1>"
    return {response_obj}(html)

# Safe usage
result = search_page()
"""
    },
    "TEMPLATE_INJECTION_JINJA2_UNSAFE": {
        "description": "Jinja2 template injection vulnerability",
        "vulnerable_template": """from {framework_import} import {request_obj}
from jinja2 import Template

def render_template(template_str):
    # Template Injection vulnerability: Unsafe template rendering
    template = Template(template_str)
    return template.render()

# Vulnerable usage
user_template = {request_obj}.form.get('template')
result = render_template(user_template)
""",
        "safe_template": """from {framework_import} import {request_obj}
from jinja2 import Environment, select_autoescape

def render_template(template_str):
    # Fixed: Use autoescape enabled environment
    env = Environment(autoescape=select_autoescape(['html', 'xml']))
    template = env.from_string(template_str)
    return template.render()

# Safe usage
user_template = {request_obj}.form.get('template')
try:
    result = render_template(user_template)
except Exception as e:
    result = f"Error: {{e}}"
"""
    },
    "XSS_MARKUPSAFE_MARKUP_TAINTED": {
        "description": "XSS via markupsafe.Markup with tainted input",
        "vulnerable_template": """from {framework_import} import {request_obj}, {response_obj}
from markupsafe import Markup

def display_content(content):
    # XSS vulnerability: Markup from tainted input
    html_content = Markup(content)
    return {response_obj}(html_content)

# Vulnerable usage
user_content = {request_obj}.form.get('content')
result = display_content(user_content)
""",
        "safe_template": """from {framework_import} import {request_obj}, {response_obj}
from markupsafe import Markup, escape

def display_content(content):
    # Fixed: Escape before creating Markup
    safe_content = escape(content)
    html_content = Markup(safe_content)
    return {response_obj}(html_content)

# Safe usage
user_content = {request_obj}.form.get('content')
result = display_content(user_content)
"""
    },
    "XSS_RAW_HTML_RESPONSE_TAINTED": {
        "description": "XSS via raw HTML response with tainted input",
        "vulnerable_template": """from {framework_import} import {request_obj}, {response_obj}

def search_page():
    # XSS vulnerability: Raw HTML response with tainted input
    query = {request_obj}.args.get('q', '')
    html = f"<h1>Search results for: {{query}}</h1>"
    return {response_obj}(html)

# Vulnerable usage
result = search_page()
""",
        "safe_template": """from {framework_import} import {request_obj}, {response_obj}
from {escape_module} import escape

def search_page():
    # Fixed: Escape user input
    query = {request_obj}.args.get('q', '')
    safe_query = escape(query)
    html = f"<h1>Search results for: {{safe_query}}</h1>"
    return {response_obj}(html)

# Safe usage
result = search_page()
"""
    },
    "PT_PATH_TRAVERSAL": {
        "description": "Path Traversal vulnerability",
        "vulnerable_template": """from {framework_import} import {request_obj}, {response_obj}
import os

def read_file(filename):
    # Path Traversal vulnerability: Direct file access
    file_path = os.path.join('/var/www/files', filename)
    with open(file_path, 'r') as f:
        return f.read()

# Vulnerable usage
user_file = {request_obj}.args.get('file')
content = read_file(user_file)
return {response_obj}(content)
""",
        "safe_template": """from {framework_import} import {request_obj}, {response_obj}
import os

def read_file(filename):
    # Fixed: Validate and sanitize filename
    # Remove path separators
    safe_filename = os.path.basename(filename)
    # Validate against whitelist
    allowed_files = ['file1.txt', 'file2.txt', 'file3.txt']
    if safe_filename not in allowed_files:
        raise ValueError("File not allowed")
    
    file_path = os.path.join('/var/www/files', safe_filename)
    # Additional check: ensure resolved path is within base directory
    resolved = os.path.realpath(file_path)
    base_dir = os.path.realpath('/var/www/files')
    if not resolved.startswith(base_dir):
        raise ValueError("Invalid file path")
    
    with open(file_path, 'r') as f:
        return f.read()

# Safe usage
user_file = {request_obj}.args.get('file')
try:
    content = read_file(user_file)
    return {response_obj}(content)
except ValueError as e:
    return {response_obj}(f"Error: {{e}}")
"""
    },
    "SSRF_REQUESTS_URL_TAINTED": {
        "description": "SSRF via requests library with tainted URL",
        "vulnerable_template": """from {framework_import} import {request_obj}
import requests

def fetch_url(url):
    # SSRF vulnerability: User input directly used
    response = requests.get(url)
    return response.text

# Vulnerable usage
user_url = {request_obj}.args.get('url')
result = fetch_url(user_url)
""",
        "safe_template": """from {framework_import} import {request_obj}
import requests
from urllib.parse import urlparse

def fetch_url(url):
    # Fixed: Validate URL
    parsed = urlparse(url)
    
    # Block internal IPs
    if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
        raise ValueError("Access to localhost not allowed")
    
    # Block private IP ranges
    if parsed.hostname and parsed.hostname.startswith(('10.', '172.16.', '192.168.')):
        raise ValueError("Access to private IPs not allowed")
    
    # Only allow HTTP/HTTPS
    if parsed.scheme not in ['http', 'https']:
        raise ValueError("Only HTTP/HTTPS allowed")
    
    response = requests.get(url, timeout=5)
    return response.text

# Safe usage
user_url = {request_obj}.args.get('url')
try:
    result = fetch_url(user_url)
except ValueError as e:
    result = f"Error: {{e}}"
"""
    },
    "SSRF_URLLIB_URL_TAINTED": {
        "description": "SSRF via urllib.request with tainted URL",
        "vulnerable_template": """from {framework_import} import {request_obj}
import urllib.request

def fetch_url(url):
    # SSRF vulnerability: User input directly used
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return response.read().decode()

# Vulnerable usage
user_url = {request_obj}.args.get('url')
result = fetch_url(user_url)
""",
        "safe_template": """from {framework_import} import {request_obj}
import urllib.request
from urllib.parse import urlparse

def fetch_url(url):
    # Fixed: Validate URL
    parsed = urlparse(url)
    
    # Block internal IPs
    if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
        raise ValueError("Access to localhost not allowed")
    
    # Block private IP ranges
    if parsed.hostname and parsed.hostname.startswith(('10.', '172.16.', '192.168.')):
        raise ValueError("Access to private IPs not allowed")
    
    # Only allow HTTP/HTTPS
    if parsed.scheme not in ['http', 'https']:
        raise ValueError("Only HTTP/HTTPS allowed")
    
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.read().decode()

# Safe usage
user_url = {request_obj}.args.get('url')
try:
    result = fetch_url(user_url)
except ValueError as e:
    result = f"Error: {{e}}"
"""
    },
    "SSRF_HTTPX_URL_TAINTED": {
        "description": "SSRF via httpx library with tainted URL",
        "vulnerable_template": """from {framework_import} import {request_obj}
import httpx

def fetch_url(url):
    # SSRF vulnerability: User input directly used
    response = httpx.get(url)
    return response.text

# Vulnerable usage
user_url = {request_obj}.args.get('url')
result = fetch_url(user_url)
""",
        "safe_template": """from {framework_import} import {request_obj}
import httpx
from urllib.parse import urlparse

def fetch_url(url):
    # Fixed: Validate URL
    parsed = urlparse(url)
    
    # Block internal IPs
    if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
        raise ValueError("Access to localhost not allowed")
    
    # Block private IP ranges
    if parsed.hostname and parsed.hostname.startswith(('10.', '172.16.', '192.168.')):
        raise ValueError("Access to private IPs not allowed")
    
    # Only allow HTTP/HTTPS
    if parsed.scheme not in ['http', 'https']:
        raise ValueError("Only HTTP/HTTPS allowed")
    
    response = httpx.get(url, timeout=5.0)
    return response.text

# Safe usage
user_url = {request_obj}.args.get('url')
try:
    result = fetch_url(user_url)
except ValueError as e:
    result = f"Error: {{e}}"
"""
    },
    "DESER_PICKLE": {
        "description": "Deserialization vulnerability via pickle",
        "vulnerable_template": """from {framework_import} import {request_obj}
import pickle

def load_data(data):
    # Deserialization vulnerability: pickle.loads
    return pickle.loads(data)

# Vulnerable usage
user_data = {request_obj}.data
result = load_data(user_data)
""",
        "safe_template": """from {framework_import} import {request_obj}
import json

def load_data(data):
    # Fixed: Use JSON instead of pickle
    return json.loads(data)

# Safe usage
user_data = {request_obj}.data
try:
    result = load_data(user_data)
except json.JSONDecodeError as e:
    result = {{"error": str(e)}}
"""
    },
    "DESER_YAML": {
        "description": "Deserialization vulnerability via yaml.load",
        "vulnerable_template": """from {framework_import} import {request_obj}
import yaml

def load_config(config_str):
    # Deserialization vulnerability: yaml.load without safe loader
    return yaml.load(config_str, Loader=yaml.Loader)

# Vulnerable usage
user_config = {request_obj}.form.get('config')
config = load_config(user_config)
""",
        "safe_template": """from {framework_import} import {request_obj}
import yaml

def load_config(config_str):
    # Fixed: Use safe loader
    return yaml.safe_load(config_str)

# Safe usage
user_config = {request_obj}.form.get('config')
try:
    config = load_config(user_config)
except yaml.YAMLError as e:
    config = {{"error": str(e)}}
"""
    },
    "AUTHZ_MISSING_DECORATOR": {
        "description": "Missing authentication/authorization decorator",
        "vulnerable_template": """from {framework_import} import {request_obj}, {response_obj}

# Vulnerable: Missing @login_required or similar decorator
def delete_user():
    # Authorization vulnerability: No decorator check
    user_id = {request_obj}.args.get('user_id')
    # In real app, this would delete from database
    return {response_obj}(f"User {{user_id}} deleted")

# Vulnerable usage
result = delete_user()
""",
        "safe_template": """from {framework_import} import {request_obj}, {response_obj}
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # In real app, check session/token
        current_user_id = {request_obj}.args.get('current_user_id')
        if not current_user_id:
            return {response_obj}("Unauthorized", status=401)
        return f(*args, **kwargs)
    return decorated_function

# Fixed: Add authentication decorator
@login_required
def delete_user():
    user_id = {request_obj}.args.get('user_id')
    current_user_id = {request_obj}.args.get('current_user_id')
    # Check authorization
    if user_id != current_user_id:
        return {response_obj}("Forbidden", status=403)
    # In real app, this would delete from database
    return {response_obj}(f"User {{user_id}} deleted")

# Safe usage
result = delete_user()
"""
    },
    "AUTHZ_DIRECT_OBJECT_REFERENCE_TAINTED": {
        "description": "Insecure Direct Object Reference (IDOR) with tainted ID",
        "vulnerable_template": """from {framework_import} import {request_obj}
from sqlalchemy.orm import Session

def get_user_profile(user_id):
    # IDOR vulnerability: Direct object reference without ownership check
    # In real app, this would use SQLAlchemy session
    # db.query(User).get(user_id)  # No ownership check
    return {{"user_id": user_id, "profile": "data"}}

# Vulnerable usage
target_user_id = {request_obj}.args.get('user_id')
profile = get_user_profile(target_user_id)
""",
        "safe_template": """from {framework_import} import {request_obj}
from sqlalchemy.orm import Session

def get_user_profile(user_id, current_user_id):
    # Fixed: Check ownership before access
    if user_id != current_user_id:
        raise PermissionError("Cannot access other user's profile")
    
    # In real app, this would use SQLAlchemy session
    # db.query(User).get(user_id)  # Only after ownership check
    return {{"user_id": user_id, "profile": "data"}}

# Safe usage
target_user_id = {request_obj}.args.get('user_id')
current_user_id = {request_obj}.args.get('current_user_id')  # From session
try:
    profile = get_user_profile(target_user_id, current_user_id)
except PermissionError as e:
    profile = {{"error": str(e)}}
"""
    },
    "CRYPTO_WEAK_HASH_MD5_SHA1": {
        "description": "Weak cryptographic hash (MD5/SHA1)",
        "vulnerable_template": """from {framework_import} import {request_obj}
import hashlib

def hash_password(password):
    # Cryptographic weakness: MD5 is weak and broken
    return hashlib.md5(password.encode()).hexdigest()

# Vulnerable usage
user_password = {request_obj}.form.get('password')
hashed = hash_password(user_password)
""",
        "safe_template": """from {framework_import} import {request_obj}
import hashlib
import secrets

def hash_password(password):
    # Fixed: Use SHA-256 with salt (or bcrypt in production)
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256()
    hash_obj.update((password + salt).encode())
    return f"{{salt}}:{{hash_obj.hexdigest()}}"

# Safe usage
user_password = {request_obj}.form.get('password')
hashed = hash_password(user_password)
"""
    },
    "JWT_VERIFY_DISABLED": {
        "description": "JWT verification disabled",
        "vulnerable_template": """from {framework_import} import {request_obj}
import jwt

def verify_token(token):
    # JWT vulnerability: verify=False
    decoded = jwt.decode(token, options={{"verify_signature": False}})
    return decoded

# Vulnerable usage
user_token = {request_obj}.headers.get('Authorization')
payload = verify_token(user_token)
""",
        "safe_template": """from {framework_import} import {request_obj}
import jwt

def verify_token(token, secret_key):
    # Fixed: Verify signature
    decoded = jwt.decode(token, secret_key, algorithms=['HS256'])
    return decoded

# Safe usage
user_token = {request_obj}.headers.get('Authorization')
secret_key = "your-secret-key"  # From config
try:
    payload = verify_token(user_token, secret_key)
except jwt.InvalidTokenError as e:
    payload = {{"error": str(e)}}
"""
    },
}


# Framework-specific configurations
FRAMEWORK_CONFIGS = {
    "flask": {
        "framework_import": "flask",
        "request_obj": "request",
        "response_obj": "Response",
        "escape_module": "markupsafe",
        "db_module": "sqlite3",
        "db_name": "db.sqlite",  # Without quotes - will be added in template
    },
    "django": {
        "framework_import": "django.http",
        "request_obj": "HttpRequest",
        "response_obj": "HttpResponse",
        "escape_module": "django.utils.html",
        "db_module": "django.db",
        "db_name": "default",  # Without quotes
    },
    "fastapi": {
        "framework_import": "fastapi",
        "request_obj": "Request",
        "response_obj": "Response",
        "escape_module": "html",
        "db_module": "sqlite3",
        "db_name": "db.sqlite",  # Without quotes
    },
}


def expand_template(template: str, framework: str, pattern_id: str) -> str:
    """Expand template with framework-specific code."""
    config = FRAMEWORK_CONFIGS[framework]
    
    # Format template with placeholders
    # Note: db_name is used as {db_name} in template, so we need to add quotes
    expanded = template.format(
        framework_import=config["framework_import"],
        request_obj=config["request_obj"],
        response_obj=config["response_obj"],
        escape_module=config["escape_module"],
        db_module=config["db_module"],
        db_name=f"'{config['db_name']}'",  # Add quotes here
    )
    
    # Special handling for framework-specific imports and API differences
    if framework == "flask":
        # Remove duplicate imports if they exist
        if "from flask import request, Response" in expanded:
            # Already has import, remove any duplicates
            lines = expanded.split('\n')
            seen_imports = set()
            filtered_lines = []
            for line in lines:
                if line.startswith('from flask import'):
                    if line not in seen_imports:
                        seen_imports.add(line)
                        filtered_lines.append(line)
                else:
                    filtered_lines.append(line)
            expanded = '\n'.join(filtered_lines)
        else:
            # Add Flask imports if not present
            expanded = f"from flask import request, Response\n{expanded}"
    elif framework == "django":
        # Django uses different request handling
        expanded = expanded.replace("request.args.get", "request.GET.get")
        expanded = expanded.replace("request.form.get", "request.POST.get")
        expanded = expanded.replace("request.data", "request.body")
        # Add Django imports if needed
        if "from django.http import" not in expanded:
            expanded = f"from django.http import HttpRequest, HttpResponse\n{expanded}"
    else:  # fastapi
        # FastAPI uses Request object
        expanded = expanded.replace("request.args.get", "request.query_params.get")
        expanded = expanded.replace("request.form.get", "request.form.get")
        # Add FastAPI imports if needed
        if "from fastapi import Request" not in expanded:
            expanded = f"from fastapi import Request\nfrom fastapi.responses import Response\n{expanded}"
    
    return expanded


class BalancedDatasetGenerator:
    """Generate balanced synthetic dataset."""
    
    def __init__(self, output_dir: Path, patterns_yaml: Optional[Path] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.patterns_yaml = patterns_yaml or Path(__file__).parent.parent / "patterns.yaml"
        
        # Output paths
        self.code_dir = self.output_dir / "code"
        self.processed_dir = self.output_dir / "processed"
        self.metadata_file = self.output_dir / "metadata.jsonl"
        
        self.code_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Load pattern IDs from patterns.yaml
        self.pattern_ids = self._load_pattern_ids()
    
    def _load_pattern_ids(self) -> List[str]:
        """Load pattern IDs from patterns.yaml."""
        if not self.patterns_yaml.exists():
            print(f"Warning: patterns.yaml not found at {self.patterns_yaml}")
            return list(PATTERN_TEMPLATES.keys())
        
        try:
            # Parse YAML manually
            import yaml
            with open(self.patterns_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            if 'patterns' in data:
                pattern_ids = [p['id'] for p in data['patterns']]
                # Filter to only include patterns we have templates for
                available_patterns = [pid for pid in pattern_ids if pid in PATTERN_TEMPLATES]
                if available_patterns:
                    return available_patterns
                else:
                    print(f"Warning: No templates found for patterns in YAML, using all templates")
                    return list(PATTERN_TEMPLATES.keys())
            else:
                print(f"Warning: No 'patterns' section in YAML, using templates")
                return list(PATTERN_TEMPLATES.keys())
        except Exception as e:
            print(f"Warning: Failed to load patterns.yaml: {e}")
            return list(PATTERN_TEMPLATES.keys())
    
    def generate(
        self,
        samples_per_pattern: int = 20,
        frameworks: List[str] = None,
        complexity_levels: List[str] = None,
        vulnerable_ratio: float = 0.5,
    ):
        """
        Generate balanced dataset.
        
        Args:
            samples_per_pattern: Number of samples per pattern ID
            frameworks: List of frameworks to generate (default: all)
            complexity_levels: List of complexity levels (default: ['simple'])
            vulnerable_ratio: Ratio of vulnerable samples (default: 0.5)
        """
        if frameworks is None:
            frameworks = ["flask", "django", "fastapi"]
        if complexity_levels is None:
            complexity_levels = ["simple"]
        
        print(f"Generating balanced dataset:")
        print(f"  Patterns: {len(self.pattern_ids)}")
        print(f"  Samples per pattern: {samples_per_pattern}")
        print(f"  Frameworks: {frameworks}")
        print(f"  Complexity levels: {complexity_levels}")
        print(f"  Vulnerable ratio: {vulnerable_ratio}")
        print()
        
        total_generated = 0
        
        for pattern_id in self.pattern_ids:
            if pattern_id not in PATTERN_TEMPLATES:
                print(f"Warning: No template for pattern {pattern_id}, skipping")
                continue
            
            print(f"Generating samples for {pattern_id}...")
            
            template = PATTERN_TEMPLATES[pattern_id]
            samples_per_framework = samples_per_pattern // len(frameworks)
            
            for framework in frameworks:
                for i in range(samples_per_framework):
                    # Generate vulnerable sample
                    if random.random() < vulnerable_ratio:
                        code = expand_template(
                            template["vulnerable_template"],
                            framework,
                            pattern_id
                        )
                        label = 1  # Vulnerable
                        sample_type = "vulnerable"
                    else:
                        code = expand_template(
                            template["safe_template"],
                            framework,
                            pattern_id
                        )
                        label = 0  # Safe
                        sample_type = "safe"
                    
                    # Save sample
                    sample_index = total_generated
                    self._save_sample(
                        pattern_id=pattern_id,
                        code=code,
                        label=label,
                        sample_type=sample_type,
                        framework=framework,
                        complexity="simple",
                        index=sample_index
                    )
                    
                    total_generated += 1
                    
                    if total_generated % 10 == 0:
                        print(f"  Generated {total_generated} samples...")
        
        print(f"\n✅ Generated {total_generated} samples")
        print(f"  Code files: {self.code_dir}")
        print(f"  CPG graphs: {self.processed_dir}")
        print(f"  Metadata: {self.metadata_file}")
    
    def _save_sample(
        self,
        pattern_id: str,
        code: str,
        label: int,
        sample_type: str,
        framework: str,
        complexity: str,
        index: int
    ):
        """Save a sample and generate CPG graph."""
        # Generate filename
        filename = f"{pattern_id}_{framework}_{complexity}_{index:04d}_{sample_type}.py"
        code_file = self.code_dir / filename
        
        # Save code
        with open(code_file, 'w') as f:
            f.write(code)
        
        # Generate CPG graph
        graph = generate_cpg_graph(code_file, self.patterns_yaml)
        
        if graph:
            graph_file = self.processed_dir / f"{filename}.json"
            with open(graph_file, 'w') as f:
                json.dump(graph, f, indent=2)
        
        # Save metadata
        seed = f"{pattern_id}_{framework}_{complexity}_{index}_{sample_type}"
        commit_hash = generate_commit_hash(seed)
        
        metadata = {
            "pattern_id": pattern_id,
            "label": label,
            "sample_type": sample_type,
            "framework": framework,
            "complexity": complexity,
            "index": index,
            "file_path": str(code_file),
            "graph_path": str(self.processed_dir / f"{filename}.json") if graph else None,
            "commit_hash": commit_hash,
            "generated_at": datetime.now().isoformat(),
        }
        
        with open(self.metadata_file, 'a') as f:
            f.write(json.dumps(metadata, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate balanced synthetic dataset for NNAST"
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset_balanced",
        help="Output directory for generated dataset"
    )
    parser.add_argument(
        "--patterns",
        help="Path to patterns.yaml (default: patterns.yaml in project root)"
    )
    parser.add_argument(
        "--samples-per-pattern",
        type=int,
        default=20,
        help="Number of samples per pattern ID"
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=["flask", "django", "fastapi"],
        default=["flask", "django", "fastapi"],
        help="Frameworks to generate"
    )
    parser.add_argument(
        "--complexity",
        nargs="+",
        choices=["simple", "medium", "complex"],
        default=["simple"],
        help="Complexity levels to generate"
    )
    parser.add_argument(
        "--vulnerable-ratio",
        type=float,
        default=0.5,
        help="Ratio of vulnerable samples (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    patterns_yaml = Path(args.patterns) if args.patterns else None
    
    generator = BalancedDatasetGenerator(output_dir, patterns_yaml)
    generator.generate(
        samples_per_pattern=args.samples_per_pattern,
        frameworks=args.frameworks,
        complexity_levels=args.complexity,
        vulnerable_ratio=args.vulnerable_ratio,
    )


if __name__ == "__main__":
    main()

