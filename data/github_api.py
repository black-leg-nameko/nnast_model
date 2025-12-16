"""
GitHub API client for collecting Python vulnerability data.

This module provides functionality to:
1. Search for CVE-related commits in Python repositories
2. Extract commit information and code diffs
3. Parse CVE/CWE information from commit messages
"""
import os
import time
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
from urllib.parse import quote


@dataclass
class GitHubCommit:
    """Represents a GitHub commit with vulnerability information."""
    sha: str
    message: str
    author: str
    date: str
    repo_url: str
    repo_name: str
    files_changed: List[str]
    cve_id: Optional[str] = None
    cwe_id: Optional[str] = None
    vulnerability_type: Optional[str] = None


@dataclass
class CodeDiff:
    """Represents code changes in a commit."""
    file_path: str
    additions: int
    deletions: int
    patch: str
    code_before: Optional[str] = None
    code_after: Optional[str] = None


class GitHubAPIClient:
    """GitHub API client with rate limiting and error handling."""
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize GitHub API client.
        
        Args:
            token: GitHub Personal Access Token (or from GITHUB_TOKEN env var)
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        
        if not self.token:
            print("Warning: GITHUB_TOKEN not set. API requests will be rate-limited.")
            print("Set GITHUB_TOKEN environment variable or create .env file for better rate limits.")
        self.session = requests.Session()
        
        if self.token:
            self.session.headers.update({
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            })
        
        self.rate_limit_remaining = 5000
        self.rate_limit_reset = 0
        self.last_request_time = 0
    
    def _check_rate_limit(self):
        """Check and handle rate limiting."""
        # Check rate limit status
        if self.rate_limit_remaining < 10:
            reset_time = self.rate_limit_reset
            current_time = time.time()
            if reset_time > current_time:
                wait_time = reset_time - current_time + 1
                print(f"Rate limit reached. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time)
        
        # Respect rate limit: max 5000 requests/hour for authenticated users
        # Add small delay between requests to avoid hitting limits
        current_time = time.time()
        if current_time - self.last_request_time < 0.1:
            time.sleep(0.1)
        self.last_request_time = current_time
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a GitHub API request with error handling.
        
        Args:
            endpoint: API endpoint (e.g., "/search/commits")
            params: Query parameters
            
        Returns:
            JSON response as dict, or None on error
        """
        self._check_rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            # Update rate limit info
            self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 5000))
            self.rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", 0))
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 403:
                if "rate limit" in response.text.lower():
                    print("Rate limit exceeded. Waiting...")
                    reset_time = self.rate_limit_reset
                    current_time = time.time()
                    if reset_time > current_time:
                        wait_time = reset_time - current_time + 1
                        time.sleep(wait_time)
                        return self._make_request(endpoint, params)
                else:
                    print(f"API error (403): {response.text}")
                    return None
            elif response.status_code == 404:
                print(f"Not found: {endpoint}")
                return None
            else:
                print(f"API error ({response.status_code}): {response.text[:200]}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
    
    def search_commits(
        self,
        query: str,
        language: str = "python",
        limit: int = 100,
        sort: str = "updated"
    ) -> List[GitHubCommit]:
        """
        Search for commits matching a query.
        
        Args:
            query: Search query (e.g., "CVE-2023")
            language: Programming language filter
            limit: Maximum number of results
            sort: Sort order ("updated", "created", "relevance")
            
        Returns:
            List of GitHubCommit objects
        """
        print(f"Searching GitHub commits: {query} (language: {language}, limit: {limit})")
        
        # GitHub search API: https://docs.github.com/en/rest/search/search#search-commits
        # Note: GitHub search API has limitations, so we'll use a combination of approaches
        
        commits = []
        per_page = min(100, limit)  # GitHub API max is 100 per page
        page = 1
        
        # Build search query
        search_query = f"{query} language:{language}"
        
        while len(commits) < limit:
            params = {
                "q": search_query,
                "sort": sort,
                "order": "desc",
                "per_page": per_page,
                "page": page
            }
            
            response = self._make_request("/search/commits", params)
            
            if not response or "items" not in response:
                break
            
            items = response["items"]
            if not items:
                break
            
            for item in items:
                if len(commits) >= limit:
                    break
                
                commit = self._parse_commit(item)
                if commit:
                    commits.append(commit)
            
            # Check if there are more pages
            if len(items) < per_page:
                break
            
            page += 1
            
            # Small delay between pages
            time.sleep(0.5)
        
        print(f"Found {len(commits)} commits")
        return commits
    
    def _parse_commit(self, item: Dict) -> Optional[GitHubCommit]:
        """Parse commit information from API response."""
        try:
            sha = item["sha"]
            commit_info = item["commit"]
            message = commit_info["message"]
            author = commit_info["author"]["name"]
            date = commit_info["author"]["date"]
            
            repo_info = item["repository"]
            repo_url = repo_info.get("html_url") or repo_info.get("url", "")
            repo_name = repo_info.get("full_name", "")
            
            # Extract CVE/CWE information from commit message
            cve_id = self._extract_cve_id(message)
            cwe_id = self._extract_cwe_id(message)
            vuln_type = self._extract_vulnerability_type(message)
            
            # Get files changed (would need additional API call)
            files_changed = []
            
            commit = GitHubCommit(
                sha=sha,
                message=message,
                author=author,
                date=date,
                repo_url=repo_url,
                repo_name=repo_name,
                files_changed=files_changed,
                cve_id=cve_id,
                cwe_id=cwe_id,
                vulnerability_type=vuln_type
            )
            
            return commit
            
        except KeyError as e:
            print(f"Error parsing commit: missing key {e}")
            return None
    
    def _extract_cve_id(self, text: str) -> Optional[str]:
        """Extract CVE ID from text (e.g., CVE-2023-1234)."""
        # Pattern: CVE-YYYY-NNNN
        pattern = r'CVE-\d{4}-\d{4,}'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return matches[0].upper() if matches else None
    
    def _extract_cwe_id(self, text: str) -> Optional[str]:
        """Extract CWE ID from text (e.g., CWE-79)."""
        # Pattern: CWE-NNN
        pattern = r'CWE-?\d{2,4}'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return matches[0].upper().replace("-", "-") if matches else None
    
    def _extract_vulnerability_type(self, text: str) -> Optional[str]:
        """Extract vulnerability type from commit message with improved detection."""
        text_lower = text.lower()
        
        # More comprehensive vulnerability type detection
        # Order matters - more specific patterns first
        
        # SQL Injection (various patterns)
        if any(pattern in text_lower for pattern in [
            "sql injection", "sqli", "sql-injection", "sql_injection",
            "sql injection vulnerability", "sql injection fix",
            "parameterized query", "prepared statement",
            "raw sql", "execute(sql", "query(sql"
        ]):
            return "SQL Injection"
        
        # XSS / Cross-Site Scripting
        if any(pattern in text_lower for pattern in [
            "xss", "cross-site scripting", "cross site scripting",
            "xss vulnerability", "xss fix", "script injection"
        ]):
            return "XSS"
        
        # Command Injection
        if any(pattern in text_lower for pattern in [
            "command injection", "command-injection", "command_injection",
            "os.system", "subprocess", "shell injection", "cmd injection",
            "arbitrary command", "command execution"
        ]):
            return "Command Injection"
        
        # Path Traversal
        if any(pattern in text_lower for pattern in [
            "path traversal", "directory traversal", "path-traversal",
            "directory-traversal", "../", "..\\", "file path",
            "arbitrary file", "local file inclusion"
        ]):
            return "Path Traversal"
        
        # XXE
        if any(pattern in text_lower for pattern in [
            "xxe", "xml external entity", "xml entity",
            "external entity injection"
        ]):
            return "XXE"
        
        # SSRF
        if any(pattern in text_lower for pattern in [
            "ssrf", "server-side request forgery", "server side request forgery",
            "request forgery", "internal request"
        ]):
            return "SSRF"
        
        # CSRF
        if any(pattern in text_lower for pattern in [
            "csrf", "cross-site request forgery", "cross site request forgery"
        ]):
            return "CSRF"
        
        # Deserialization
        if any(pattern in text_lower for pattern in [
            "deserialization", "pickle", "unsafe deserialization",
            "insecure deserialization", "yaml.load", "marshal.loads"
        ]):
            return "Deserialization"
        
        # Authentication Bypass
        if any(pattern in text_lower for pattern in [
            "authentication bypass", "auth bypass", "bypass authentication",
            "unauthorized access", "auth vulnerability"
        ]):
            return "Authentication Bypass"
        
        # Authorization Bypass
        if any(pattern in text_lower for pattern in [
            "authorization bypass", "authorization vulnerability",
            "privilege escalation", "access control", "permission bypass"
        ]):
            return "Authorization Bypass"
        
        # Information Disclosure
        if any(pattern in text_lower for pattern in [
            "information disclosure", "sensitive data exposure",
            "data leak", "information leak", "exposed credentials"
        ]):
            return "Information Disclosure"
        
        # Cryptographic Weakness
        if any(pattern in text_lower for pattern in [
            "cryptographic weakness", "weak cryptography", "weak encryption",
            "insecure random", "weak hash", "md5", "sha1", "weak crypto"
        ]):
            return "Cryptographic Weakness"
        
        # Buffer Overflow
        if any(pattern in text_lower for pattern in [
            "buffer overflow", "buffer-overflow", "stack overflow",
            "heap overflow", "integer overflow"
        ]):
            return "Buffer Overflow"
        
        # Race Condition
        if any(pattern in text_lower for pattern in [
            "race condition", "race-condition", "time-of-check",
            "toctou", "concurrent access"
        ]):
            return "Race Condition"
        
        # Hardcoded Credentials
        if any(pattern in text_lower for pattern in [
            "hardcoded", "hard-coded", "hard coded",
            "hardcoded credentials", "hardcoded secret", "hardcoded password"
        ]):
            return "Hardcoded Credentials"
        
        # Code Injection (check before other injections to avoid false positives)
        if any(pattern in text_lower for pattern in [
            "code injection", "code-injection", "code_injection",
            "code execution", "arbitrary code execution",
            "eval vulnerability", "exec vulnerability",
            "eval()", "exec()", "compile()",
            "unsafe eval", "unsafe exec"
        ]):
            return "Code Injection"
        
        # DoS (Denial of Service)
        if any(pattern in text_lower for pattern in [
            "denial of service", "dos vulnerability", "dos attack",
            "resource exhaustion", "infinite loop", "reDoS",
            "regular expression denial of service"
        ]):
            return "DoS"
        
        # Security Misconfiguration
        if any(pattern in text_lower for pattern in [
            "security misconfiguration", "misconfiguration",
            "insecure configuration", "weak configuration"
        ]):
            return "Security Misconfiguration"
        
        # Insecure Dependencies
        if any(pattern in text_lower for pattern in [
            "vulnerable dependency", "outdated dependency",
            "dependency vulnerability", "dependency update",
            "security update", "dependency bump"
        ]):
            return "Vulnerable Dependency"
        
        return None
    
    def get_commit_details(self, owner: str, repo: str, sha: str) -> Optional[Dict]:
        """
        Get detailed commit information including file changes.
        
        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            
        Returns:
            Commit details including files changed
        """
        endpoint = f"/repos/{owner}/{repo}/commits/{sha}"
        return self._make_request(endpoint)
    
    def get_file_content(self, owner: str, repo: str, path: str, ref: str = "main") -> Optional[str]:
        """
        Get file content from repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            ref: Branch/tag/commit SHA
            
        Returns:
            File content as string, or None on error
        """
        endpoint = f"/repos/{owner}/{repo}/contents/{quote(path)}"
        params = {"ref": ref}
        
        response = self._make_request(endpoint, params)
        if response and "content" in response:
            import base64
            content = base64.b64decode(response["content"]).decode("utf-8")
            return content
        
        return None
    
    def get_commit_diff(self, owner: str, repo: str, sha: str) -> List[CodeDiff]:
        """
        Get commit diff (file changes).
        
        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            
        Returns:
            List of CodeDiff objects
        """
        commit_details = self.get_commit_details(owner, repo, sha)
        if not commit_details or "files" not in commit_details:
            return []
        
        diffs = []
        for file_info in commit_details["files"]:
            if file_info.get("status") == "removed":
                continue  # Skip deleted files
            
            # Only process Python files
            if not file_info.get("filename", "").endswith(".py"):
                continue
            
            diff = CodeDiff(
                file_path=file_info.get("filename", ""),
                additions=file_info.get("additions", 0),
                deletions=file_info.get("deletions", 0),
                patch=file_info.get("patch", ""),
            )
            diffs.append(diff)
        
        return diffs
    
    def get_parent_commit(self, owner: str, repo: str, sha: str) -> Optional[str]:
        """
        Get parent commit SHA (the commit before the fix).
        
        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            
        Returns:
            Parent commit SHA, or None
        """
        commit_details = self.get_commit_details(owner, repo, sha)
        if commit_details and "parents" in commit_details:
            parents = commit_details["parents"]
            if parents:
                return parents[0]["sha"]
        return None

