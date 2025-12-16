#!/usr/bin/env python3
"""
Generate synthetic vulnerability code samples directly (without API calls).

This script generates vulnerable and fixed code pairs for specific
vulnerability types and saves them directly to the dataset directory.
"""
import argparse
import json
import os
import sys
import pathlib
import subprocess
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def generate_commit_hash(seed: str) -> str:
    """Generate a deterministic commit hash from a seed."""
    return hashlib.md5(seed.encode()).hexdigest()[:16]


def generate_cpg_graph(code_file: Path) -> Optional[Dict]:
    """Generate CPG graph from code file."""
    try:
        # Use temporary file for output
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
        try:
            Path(tmp_out_path).unlink(missing_ok=True)
        except:
            pass
    
    return None


# Synthetic code samples
SSRF_SAMPLES = [
    {
        "vulnerable": """import requests
import urllib.parse

def fetch_url(url):
    # SSRF vulnerability: user input directly used in request
    response = requests.get(url)
    return response.text

# Example usage
user_input = input("Enter URL: ")
result = fetch_url(user_input)
print(result)
""",
        "fixed": """import requests
import urllib.parse
from urllib.parse import urlparse

def fetch_url(url):
    # Fixed: Validate URL to prevent SSRF
    parsed = urlparse(url)
    
    # Block internal/private IPs
    if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
        raise ValueError("Access to localhost is not allowed")
    
    # Block private IP ranges
    if parsed.hostname and parsed.hostname.startswith(('10.', '172.16.', '192.168.')):
        raise ValueError("Access to private IP ranges is not allowed")
    
    # Only allow HTTP/HTTPS
    if parsed.scheme not in ['http', 'https']:
        raise ValueError("Only HTTP and HTTPS are allowed")
    
    response = requests.get(url, timeout=5)
    return response.text

# Example usage
user_input = input("Enter URL: ")
try:
    result = fetch_url(user_input)
    print(result)
except ValueError as e:
    print(f"Error: {e}")
""",
        "description": "SSRF vulnerability: User input directly used in HTTP request without validation, allowing access to internal services. Fixed by validating URL scheme and blocking private IP ranges."
    },
    {
        "vulnerable": """import urllib.request
import json

def proxy_request(url):
    # SSRF vulnerability: No validation of target URL
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

# Vulnerable usage
target = input("Enter target URL: ")
data = proxy_request(target)
print(data)
""",
        "fixed": """import urllib.request
import json
from urllib.parse import urlparse

ALLOWED_DOMAINS = ['api.example.com', 'public-api.example.com']

def proxy_request(url):
    # Fixed: Validate domain whitelist
    parsed = urlparse(url)
    
    if parsed.scheme not in ['http', 'https']:
        raise ValueError("Only HTTP/HTTPS allowed")
    
    if parsed.hostname not in ALLOWED_DOMAINS:
        raise ValueError(f"Domain {parsed.hostname} is not in whitelist")
    
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode())

# Fixed usage
target = input("Enter target URL: ")
try:
    data = proxy_request(target)
    print(data)
except ValueError as e:
    print(f"Error: {e}")
""",
        "description": "SSRF vulnerability: Proxy function accepts any URL without domain whitelist validation. Fixed by implementing domain whitelist and scheme validation."
    },
    {
        "vulnerable": """import requests
import os

def download_file(url, save_path):
    # SSRF vulnerability: URL used directly, can access file:// protocol
    response = requests.get(url, stream=True)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

# Vulnerable usage
user_url = input("Enter file URL: ")
download_file(user_url, "/tmp/downloaded_file")
""",
        "fixed": """import requests
import os
from urllib.parse import urlparse

def download_file(url, save_path):
    # Fixed: Validate URL scheme and hostname
    parsed = urlparse(url)
    
    # Block file:// and other dangerous schemes
    if parsed.scheme not in ['http', 'https']:
        raise ValueError(f"Scheme {parsed.scheme} is not allowed")
    
    # Block localhost and private IPs
    hostname = parsed.hostname
    if hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
        raise ValueError("Access to localhost is not allowed")
    
    if hostname and (hostname.startswith('10.') or 
                     hostname.startswith('172.16.') or 
                     hostname.startswith('192.168.')):
        raise ValueError("Access to private IP ranges is not allowed")
    
    response = requests.get(url, stream=True, timeout=10)
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return save_path

# Fixed usage
user_url = input("Enter file URL: ")
try:
    download_file(user_url, "/tmp/downloaded_file")
except ValueError as e:
    print(f"Error: {e}")
""",
        "description": "SSRF vulnerability: File download function accepts any URL including file:// protocol, allowing local file access. Fixed by validating URL scheme and blocking private IPs."
    }
]

DESERIALIZATION_SAMPLES = [
    {
        "vulnerable": """import pickle
import os

def load_user_data(filename):
    # Deserialization vulnerability: pickle.loads with untrusted data
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())
    return data

# Vulnerable usage
user_file = input("Enter data file: ")
user_data = load_user_data(user_file)
print(user_data)
""",
        "fixed": """import json
import os

def load_user_data(filename):
    # Fixed: Use JSON instead of pickle for untrusted data
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# Fixed usage
user_file = input("Enter data file: ")
try:
    user_data = load_user_data(user_file)
    print(user_data)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
""",
        "description": "Deserialization vulnerability: Using pickle.loads with untrusted input allows arbitrary code execution. Fixed by using JSON for data serialization instead of pickle."
    },
    {
        "vulnerable": """import yaml
import os

def load_config(config_file):
    # Deserialization vulnerability: yaml.load without safe loader
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)  # Dangerous!
    return config

# Vulnerable usage
config_path = input("Enter config file: ")
config = load_config(config_path)
print(config)
""",
        "fixed": """import yaml
import os

def load_config(config_file):
    # Fixed: Use safe YAML loader
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)  # Safe loader
    return config

# Fixed usage
config_path = input("Enter config file: ")
try:
    config = load_config(config_path)
    print(config)
except yaml.YAMLError as e:
    print(f"Invalid YAML: {e}")
""",
        "description": "Deserialization vulnerability: yaml.load with unsafe Loader allows arbitrary code execution via YAML payloads. Fixed by using yaml.safe_load which prevents code execution."
    },
    {
        "vulnerable": """import pickle
import marshal

def restore_session(session_data):
    # Deserialization vulnerability: Multiple unsafe deserialization methods
    if session_data.startswith(b'pickle:'):
        return pickle.loads(session_data[7:])
    elif session_data.startswith(b'marshal:'):
        return marshal.loads(session_data[8:])
    else:
        return session_data.decode()

# Vulnerable usage
data = input("Enter session data: ").encode()
session = restore_session(data)
print(session)
""",
        "fixed": """import json
import base64

def restore_session(session_data):
    # Fixed: Use safe serialization format (JSON)
    try:
        if isinstance(session_data, bytes):
            session_data = session_data.decode()
        
        # Expect JSON format
        data = json.loads(session_data)
        return data
    except json.JSONDecodeError:
        raise ValueError("Invalid session data format")

# Fixed usage
data = input("Enter session data: ")
try:
    session = restore_session(data.encode() if isinstance(data, str) else data)
    print(session)
except ValueError as e:
    print(f"Error: {e}")
""",
        "description": "Deserialization vulnerability: Using pickle and marshal for deserialization allows arbitrary code execution. Fixed by using JSON which is safe and does not execute code."
    }
]

CRYPTO_SAMPLES = [
    {
        "vulnerable": """import hashlib

def hash_password(password):
    # Cryptographic weakness: Using MD5 (broken hash function)
    return hashlib.md5(password.encode()).hexdigest()

# Vulnerable usage
user_password = input("Enter password: ")
hashed = hash_password(user_password)
print(f"Hashed password: {hashed}")
""",
        "fixed": """import hashlib
import secrets

def hash_password(password):
    # Fixed: Use bcrypt or PBKDF2 for password hashing
    # For demonstration, using SHA-256 with salt (in production, use bcrypt)
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256()
    hash_obj.update((password + salt).encode())
    return f"{salt}:{hash_obj.hexdigest()}"

def verify_password(password, stored_hash):
    salt, hash_value = stored_hash.split(':')
    hash_obj = hashlib.sha256()
    hash_obj.update((password + salt).encode())
    return hash_obj.hexdigest() == hash_value

# Fixed usage
user_password = input("Enter password: ")
hashed = hash_password(user_password)
print(f"Hashed password: {hashed}")
""",
        "description": "Cryptographic weakness: Using MD5 for password hashing is insecure as MD5 is broken and fast. Fixed by using SHA-256 with salt (or better, bcrypt/PBKDF2 in production)."
    },
    {
        "vulnerable": """import os

def generate_token():
    # Cryptographic weakness: Using os.urandom without proper randomness
    # and weak token generation
    token = os.urandom(8).hex()  # Too short and not cryptographically secure
    return token

# Vulnerable usage
api_token = generate_token()
print(f"API Token: {api_token}")
""",
        "fixed": """import secrets

def generate_token():
    # Fixed: Use secrets.token_urlsafe for cryptographically secure tokens
    token = secrets.token_urlsafe(32)  # 32 bytes = 256 bits of entropy
    return token

# Fixed usage
api_token = generate_token()
print(f"API Token: {api_token}")
""",
        "description": "Cryptographic weakness: Using os.urandom with insufficient length and not using proper secure token generation. Fixed by using secrets.token_urlsafe with adequate length for cryptographic security."
    },
    {
        "vulnerable": """import hashlib
import hmac

def verify_signature(message, signature, secret):
    # Cryptographic weakness: Timing attack vulnerability
    expected = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return signature == expected  # Vulnerable to timing attacks

# Vulnerable usage
msg = "important message"
sig = input("Enter signature: ")
secret = "my_secret_key"
if verify_signature(msg, sig, secret):
    print("Signature valid")
else:
    print("Signature invalid")
""",
        "fixed": """import hashlib
import hmac
import secrets

def verify_signature(message, signature, secret):
    # Fixed: Use hmac.compare_digest to prevent timing attacks
    expected = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)  # Constant-time comparison

# Fixed usage
msg = "important message"
sig = input("Enter signature: ")
secret = "my_secret_key"
if verify_signature(msg, sig, secret):
    print("Signature valid")
else:
    print("Signature invalid")
""",
        "description": "Cryptographic weakness: Direct string comparison in signature verification is vulnerable to timing attacks. Fixed by using hmac.compare_digest for constant-time comparison."
    }
]

XSS_SAMPLES = [
    {
        "vulnerable": """from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/search')
def search():
    # XSS vulnerability: User input directly inserted into HTML
    query = request.args.get('q', '')
    html = f"<h1>Search Results for: {query}</h1>"
    return html

if __name__ == '__main__':
    app.run()
""",
        "fixed": """from flask import Flask, request, make_response, escape

app = Flask(__name__)

@app.route('/search')
def search():
    # Fixed: Escape user input to prevent XSS
    query = request.args.get('q', '')
    escaped_query = escape(query)  # Escape HTML special characters
    html = f"<h1>Search Results for: {escaped_query}</h1>"
    return html

if __name__ == '__main__':
    app.run()
""",
        "description": "XSS vulnerability: User input directly inserted into HTML without escaping allows script injection. Fixed by escaping HTML special characters using escape() function."
    },
    {
        "vulnerable": """def render_comment(comment_text):
    # XSS vulnerability: User input directly in HTML template
    html = f'''
    <div class="comment">
        <p>{comment_text}</p>
    </div>
    '''
    return html

# Vulnerable usage
user_comment = input("Enter comment: ")
output = render_comment(user_comment)
print(output)
""",
        "fixed": """import html

def render_comment(comment_text):
    # Fixed: Escape HTML entities
    escaped_text = html.escape(comment_text)
    html_output = f'''
    <div class="comment">
        <p>{escaped_text}</p>
    </div>
    '''
    return html_output

# Fixed usage
user_comment = input("Enter comment: ")
output = render_comment(user_comment)
print(output)
""",
        "description": "XSS vulnerability: User comments directly inserted into HTML without escaping. Fixed by using html.escape() to escape HTML entities and prevent script injection."
    },
    {
        "vulnerable": """def generate_user_profile(username, bio):
    # XSS vulnerability: Multiple user inputs without sanitization
    profile_html = f'''
    <div class="profile">
        <h2>Welcome, {username}!</h2>
        <p>Bio: {bio}</p>
    </div>
    '''
    return profile_html

# Vulnerable usage
name = input("Enter username: ")
user_bio = input("Enter bio: ")
profile = generate_user_profile(name, user_bio)
print(profile)
""",
        "fixed": """import html

def generate_user_profile(username, bio):
    # Fixed: Escape all user inputs
    safe_username = html.escape(username)
    safe_bio = html.escape(bio)
    profile_html = f'''
    <div class="profile">
        <h2>Welcome, {safe_username}!</h2>
        <p>Bio: {safe_bio}</p>
    </div>
    '''
    return profile_html

# Fixed usage
name = input("Enter username: ")
user_bio = input("Enter bio: ")
profile = generate_user_profile(name, user_bio)
print(profile)
""",
        "description": "XSS vulnerability: Multiple user inputs (username, bio) directly inserted into HTML without escaping. Fixed by escaping all user inputs using html.escape()."
    },
    {
        "vulnerable": """def render_error_message(error_msg):
    # XSS vulnerability: Error message directly in HTML
    return f'<div class="error">Error: {error_msg}</div>'

# Vulnerable usage
error = request.args.get('error', 'Unknown error')
response = render_error_message(error)
""",
        "fixed": """import html

def render_error_message(error_msg):
    # Fixed: Escape error message
    safe_error = html.escape(error_msg)
    return f'<div class="error">Error: {safe_error}</div>'

# Fixed usage
error = request.args.get('error', 'Unknown error')
response = render_error_message(error)
""",
        "description": "XSS vulnerability: Error messages from URL parameters directly inserted into HTML. Fixed by escaping error messages to prevent XSS attacks."
    },
    {
        "vulnerable": """def build_url(base_url, params):
    # XSS vulnerability: URL parameters not sanitized
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    return f"{base_url}?{query_string}"

# Vulnerable usage
user_params = {'redirect': input("Enter redirect URL: ")}
url = build_url('https://example.com', user_params)
print(f"Redirecting to: {url}")
""",
        "fixed": """import urllib.parse

def build_url(base_url, params):
    # Fixed: URL encode parameters
    safe_params = {k: urllib.parse.quote(str(v), safe='') for k, v in params.items()}
    query_string = '&'.join([f"{k}={v}" for k, v in safe_params.items()])
    return f"{base_url}?{query_string}"

# Fixed usage
user_params = {'redirect': input("Enter redirect URL: ")}
url = build_url('https://example.com', user_params)
print(f"Redirecting to: {url}")
""",
        "description": "XSS vulnerability: URL parameters not properly encoded, allowing script injection in URLs. Fixed by URL encoding all parameter values."
    }
]

CODE_INJECTION_SAMPLES = [
    {
        "vulnerable": """def calculate(expression):
    # Code injection vulnerability: eval() with user input
    result = eval(expression)
    return result

# Vulnerable usage
user_expr = input("Enter expression: ")
answer = calculate(user_expr)
print(f"Result: {answer}")
""",
        "fixed": """import ast
import operator

def calculate(expression):
    # Fixed: Use ast.literal_eval for safe evaluation
    # Only allows literals, not arbitrary code execution
    try:
        # For simple arithmetic, use ast.parse and evaluate safely
        node = ast.parse(expression, mode='eval')
        if isinstance(node.body, (ast.Constant, ast.Num)):
            return node.body.value
        # For more complex expressions, use a safe evaluator
        return ast.literal_eval(expression)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid expression")

# Fixed usage
user_expr = input("Enter expression: ")
try:
    answer = calculate(user_expr)
    print(f"Result: {answer}")
except ValueError as e:
    print(f"Error: {e}")
""",
        "description": "Code injection vulnerability: Using eval() with user input allows arbitrary code execution. Fixed by using ast.literal_eval() which only evaluates literals safely."
    },
    {
        "vulnerable": """import subprocess

def run_command(cmd):
    # Code injection vulnerability: User input directly in shell command
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

# Vulnerable usage
user_cmd = input("Enter command: ")
output = run_command(user_cmd)
print(output)
""",
        "fixed": """import subprocess
import shlex

def run_command(cmd):
    # Fixed: Use subprocess without shell=True and validate input
    # Split command safely and execute without shell
    if not cmd or not cmd.strip():
        raise ValueError("Empty command")
    
    # Use shlex.split to safely parse command
    cmd_parts = shlex.split(cmd)
    
    # Whitelist allowed commands (example)
    ALLOWED_COMMANDS = ['ls', 'pwd', 'date']
    if cmd_parts[0] not in ALLOWED_COMMANDS:
        raise ValueError(f"Command {cmd_parts[0]} not allowed")
    
    result = subprocess.run(cmd_parts, shell=False, capture_output=True, text=True)
    return result.stdout

# Fixed usage
user_cmd = input("Enter command: ")
try:
    output = run_command(user_cmd)
    print(output)
except ValueError as e:
    print(f"Error: {e}")
""",
        "description": "Code injection vulnerability: Using subprocess with shell=True and user input allows command injection. Fixed by using shell=False, shlex.split, and command whitelisting."
    },
    {
        "vulnerable": """def execute_python_code(code_string):
    # Code injection vulnerability: exec() with user input
    exec(code_string)
    return "Code executed"

# Vulnerable usage
user_code = input("Enter Python code: ")
result = execute_python_code(user_code)
print(result)
""",
        "fixed": """import ast

def execute_python_code(code_string):
    # Fixed: Parse and validate code structure without executing
    # This is a safe alternative that doesn't execute arbitrary code
    try:
        tree = ast.parse(code_string)
        # Only allow certain AST node types (example: expressions only)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
                # Block dangerous operations
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', '__import__']:
                        raise ValueError("Dangerous function calls not allowed")
        return "Code validated (not executed for safety)"
    except SyntaxError:
        raise ValueError("Invalid Python syntax")

# Fixed usage
user_code = input("Enter Python code: ")
try:
    result = execute_python_code(user_code)
    print(result)
except ValueError as e:
    print(f"Error: {e}")
""",
        "description": "Code injection vulnerability: Using exec() with user input allows arbitrary code execution. Fixed by parsing code with ast.parse and validating structure without executing."
    },
    {
        "vulnerable": """import os

def process_file(filename):
    # Code injection vulnerability: os.system with user input
    os.system(f"cat {filename}")

# Vulnerable usage
user_file = input("Enter filename: ")
process_file(user_file)
""",
        "fixed": """def process_file(filename):
    # Fixed: Read file directly without shell command
    import os.path
    import os
    
    # Validate filename to prevent path traversal
    if '..' in filename or os.sep in filename:
        raise ValueError("Invalid filename")
    
    # Read file directly
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
        return content
    else:
        raise FileNotFoundError(f"File {filename} not found")

# Fixed usage
user_file = input("Enter filename: ")
try:
    content = process_file(user_file)
    print(content)
except (ValueError, FileNotFoundError) as e:
    print(f"Error: {e}")
""",
        "description": "Code injection vulnerability: Using os.system() with user input allows command injection. Fixed by reading files directly without shell commands and validating filenames."
    },
    {
        "vulnerable": """def template_render(template_str, **kwargs):
    # Code injection vulnerability: String formatting with exec-like behavior
    return template_str.format(**kwargs)

# Vulnerable usage
user_template = input("Enter template: ")
result = template_render(user_template, name="User")
print(result)
""",
        "fixed": """def template_render(template_str, **kwargs):
    # Fixed: Use safe string formatting with validation
    # Only allow simple variable substitution
    try:
        # Validate template doesn't contain dangerous patterns
        if '{' in template_str and '}' in template_str:
            # Check for nested expressions or function calls
            if '__' in template_str or '[' in template_str or '(' in template_str:
                raise ValueError("Complex expressions not allowed in template")
        
        # Use safe formatting
        return template_str.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}")

# Fixed usage
user_template = input("Enter template: ")
try:
    result = template_render(user_template, name="User")
    print(result)
except ValueError as e:
    print(f"Error: {e}")
""",
        "description": "Code injection vulnerability: Template string formatting with user input can allow code execution. Fixed by validating template structure and preventing complex expressions."
    }
]


def save_synthetic_sample(
    dataset_dir: Path,
    vulnerability_type: str,
    cwe_id: str,
    sample: Dict,
    index: int
):
    """Save a synthetic vulnerability sample to the dataset."""
    code_dir = dataset_dir / "code"
    processed_dir = dataset_dir / "processed"
    metadata_file = dataset_dir / "metadata.jsonl"
    
    code_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    # Generate deterministic commit hashes
    seed = f"{vulnerability_type}_{index}_{sample['description'][:20]}"
    commit_before = generate_commit_hash(f"{seed}_before")
    commit_after = generate_commit_hash(f"{seed}_after")
    
    # Generate filename
    filename = f"{vulnerability_type.lower()}_sample_{index}.py"
    
    # Save code files
    code_before_file = code_dir / f"{commit_before}_{filename}_before.py"
    code_after_file = code_dir / f"{commit_after}_{filename}_after.py"
    
    code_before_file.write_text(sample["vulnerable"], encoding="utf-8")
    code_after_file.write_text(sample["fixed"], encoding="utf-8")
    
    print(f"  Saved code files: {code_before_file.name}, {code_after_file.name}")
    
    # Generate CPG graphs
    print(f"  Generating CPG graphs...")
    graph_before = generate_cpg_graph(code_before_file)
    graph_after = generate_cpg_graph(code_after_file)
    
    # Save CPG graphs
    if graph_before:
        graph_file_before = processed_dir / f"{commit_before}_{Path(filename).stem}_before.jsonl"
        with open(graph_file_before, "w") as f:
            f.write(json.dumps(graph_before, ensure_ascii=False) + "\n")
        print(f"  Saved CPG: {graph_file_before.name}")
    
    if graph_after:
        graph_file_after = processed_dir / f"{commit_after}_{Path(filename).stem}_after.jsonl"
        with open(graph_file_after, "w") as f:
            f.write(json.dumps(graph_after, ensure_ascii=False) + "\n")
        print(f"  Saved CPG: {graph_file_after.name}")
    
    # Save metadata
    metadata = {
        "cve_id": None,
        "cwe_id": cwe_id,
        "repo_url": None,
        "commit_before": commit_before,
        "commit_after": commit_after,
        "file_path": filename,
        "vulnerability_type": vulnerability_type,
        "description": sample["description"],
        "code_before_file": str(code_before_file.relative_to(dataset_dir)),
        "code_after_file": str(code_after_file.relative_to(dataset_dir)),
        "timestamp": datetime.now().isoformat(),
        "synthetic": True
    }
    
    with open(metadata_file, "a") as f:
        f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    
    print(f"  Added metadata entry")
    return metadata


def get_current_counts(dataset_dir: Path) -> Dict[str, int]:
    """Get current count of each vulnerability type in dataset."""
    metadata_file = dataset_dir / "metadata.jsonl"
    if not metadata_file.exists():
        return {}
    
    counts = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                vtype = record.get('vulnerability_type')
                if vtype:
                    counts[vtype] = counts.get(vtype, 0) + 1
            except:
                continue
    return counts


def calculate_needed(targets: Dict[str, int], current_counts: Dict[str, int]) -> Dict[str, int]:
    """Calculate how many samples are needed for each type."""
    needed = {}
    for vtype, target in targets.items():
        current = current_counts.get(vtype, 0)
        needed[vtype] = max(0, target - current)
    return needed


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic vulnerability code samples directly"
    )
    parser.add_argument(
        "--dataset-dir",
        default="./dataset",
        help="Dataset directory (default: ./dataset)"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["SSRF", "Deserialization", "Cryptographic Weakness", "XSS", "Code Injection"],
        default=None,
        help="Vulnerability types to generate (default: all types with targets)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of samples per type (default: auto-calculate to reach target)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        help="JSON string with target counts per type (e.g., '{\"SSRF\": 20, \"XSS\": 20}')"
    )
    parser.add_argument(
        "--show-status",
        action="store_true",
        help="Show current dataset status and exit"
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(exist_ok=True)
    
    # Default target counts
    default_targets = {
        "SSRF": 20,
        "Deserialization": 20,
        "Cryptographic Weakness": 15,
        "XSS": 20,
        "Code Injection": 15
    }
    
    # Parse custom targets if provided
    if args.targets:
        import json
        custom_targets = json.loads(args.targets)
        default_targets.update(custom_targets)
    
    targets = default_targets
    
    # Get current counts
    current_counts = get_current_counts(dataset_dir)
    
    # Show status if requested
    if args.show_status:
        print("\n=== 現在のデータセット状況 ===")
        print(f"総レコード数: {sum(current_counts.values())}")
        print("\n脆弱性タイプ別の内訳:")
        for vtype in sorted(set(list(targets.keys()) + list(current_counts.keys()))):
            current = current_counts.get(vtype, 0)
            target = targets.get(vtype, 0)
            needed = max(0, target - current)
            status = "✓" if current >= target else "⚠" if current > 0 else "✗"
            print(f"  {status} {vtype}: {current}件 / 目標{target}件 (あと{needed}件必要)")
        return 0
    
    # Determine which types to generate
    if args.types is None:
        # Auto-select types that need more samples
        needed = calculate_needed(targets, current_counts)
        types_to_generate = [vtype for vtype, count in needed.items() if count > 0]
        if not types_to_generate:
            print("All vulnerability types have reached their targets!")
            return 0
    else:
        types_to_generate = args.types
    
    # Sample collections
    samples_map = {
        "SSRF": SSRF_SAMPLES,
        "Deserialization": DESERIALIZATION_SAMPLES,
        "Cryptographic Weakness": CRYPTO_SAMPLES,
        "XSS": XSS_SAMPLES,
        "Code Injection": CODE_INJECTION_SAMPLES
    }
    
    # CWE IDs
    cwe_map = {
        "SSRF": "CWE-918",
        "Deserialization": "CWE-502",
        "Cryptographic Weakness": "CWE-327",
        "XSS": "CWE-79",
        "Code Injection": "CWE-94"
    }
    
    total_generated = 0
    
    for vuln_type in types_to_generate:
        if vuln_type not in samples_map:
            print(f"Warning: No samples defined for {vuln_type}")
            continue
        
        if vuln_type not in samples_map:
            print(f"Warning: No samples defined for {vuln_type}")
            continue
        
        samples = samples_map[vuln_type]
        cwe_id = cwe_map.get(vuln_type)
        
        # Determine count
        if args.count is not None:
            count = min(args.count, len(samples))
        else:
            # Auto-calculate: generate enough to reach target, but don't exceed available samples
            current = current_counts.get(vuln_type, 0)
            target = targets.get(vuln_type, 0)
            needed = max(0, target - current)
            count = min(needed, len(samples))
            
            if count == 0:
                print(f"\n{vuln_type}: Already at target ({current}/{target}), skipping...")
                continue
        
        print(f"\n{'='*50}")
        print(f"Generating {count} {vuln_type} samples")
        if args.count is None:
            current = current_counts.get(vuln_type, 0)
            target = targets.get(vuln_type, 0)
            print(f"Current: {current}, Target: {target}, Needed: {count}")
        print(f"{'='*50}")
        
        for i in range(count):
            print(f"\n[{i+1}/{count}] {vuln_type} sample {i+1}")
            try:
                # Use modulo to cycle through samples if count > len(samples)
                sample_index = i % len(samples)
                # Use a unique index for commit hash generation
                unique_index = current_counts.get(vuln_type, 0) + i
                metadata = save_synthetic_sample(
                    dataset_dir,
                    vuln_type,
                    cwe_id,
                    samples[sample_index],
                    unique_index
                )
                total_generated += 1
            except Exception as e:
                print(f"  Error generating sample: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"Generation complete!")
    print(f"Total samples generated: {total_generated}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

