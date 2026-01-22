#!/usr/bin/env python3
"""
GitHub Issue creation functionality

Creates/updates reports as GitHub Issues based on the design document.
"""
import os
import hashlib
from typing import Dict, List, Any, Optional
from github import Github


class GitHubIssueManager:
    """GitHub Issue management class"""
    
    def __init__(self, token: str, repo: str):
        """
        Args:
            token: GitHub Personal Access Token (GITHUB_TOKEN)
            repo: Repository (org/repo format)
        """
        self.github = Github(token)
        self.repo = self.github.get_repo(repo)
        self.repo_name = repo
    
    def find_existing_issue(self, fingerprint: str) -> Optional[Any]:
        """
        Search for existing Issue
        
        Args:
            fingerprint: Issue fingerprint
            
        Returns:
            Found Issue, or None
        """
        # Search by label
        label = "nnast"
        query = f'repo:{self.repo_name} label:{label}'
        
        issues = self.github.search_issues(query)
        
        for issue in issues:
            # Extract fingerprint from issue body
            body = issue.body or ""
            if f"fp:{fingerprint}" in body:
                return issue
        
        return None
    
    def create_issue(
        self,
        title: str,
        markdown: str,
        labels: List[str],
        fingerprint: str
    ) -> Any:
        """
        Create Issue
        
        Args:
            title: Issue title
            markdown: Issue body (Markdown)
            labels: Label list
            fingerprint: Issue fingerprint
            
        Returns:
            Created Issue
        """
        # Add fingerprint to body
        body_with_fp = f"{markdown}\n\n---\n\n<!-- fp:{fingerprint} -->"
        
        issue = self.repo.create_issue(
            title=title,
            body=body_with_fp,
            labels=labels
        )
        
        return issue
    
    def update_issue(
        self,
        issue: Any,
        markdown: str,
        commit_sha: Optional[str] = None
    ) -> Any:
        """
        Add comment to existing Issue
        
        Args:
            issue: GitHub Issue object
            markdown: Markdown to add
            commit_sha: Commit SHA (optional)
            
        Returns:
            Updated Issue
        """
        comment_body = markdown
        if commit_sha:
            comment_body = f"## New Evidence (Commit: {commit_sha[:8]})\n\n{markdown}"
        
        issue.create_comment(comment_body)
        
        return issue
    
    def create_or_update_issue(
        self,
        report: Dict[str, Any],
        commit_sha: Optional[str] = None
    ) -> Any:
        """
        Create or update Issue
        
        Args:
            report: Report information
            commit_sha: Commit SHA (optional)
            
        Returns:
            Issue object
        """
        fingerprint = report.get("fingerprint")
        title = report.get("title", "NNAST Security Finding")
        markdown = report.get("markdown", "")
        labels = report.get("labels", [])
        
        if not fingerprint:
            raise ValueError("fingerprint is required")
        
        # Search for existing Issue
        existing_issue = self.find_existing_issue(fingerprint)
        
        if existing_issue:
            # Add comment to existing Issue
            return self.update_issue(existing_issue, markdown, commit_sha)
        else:
            # Create new Issue
            return self.create_issue(title, markdown, labels, fingerprint)


def create_issues_from_reports(
    reports: List[Dict[str, Any]],
    github_token: str,
    repo: str,
    commit_sha: Optional[str] = None,
    max_issues: int = 5
) -> List[Any]:
    """
    Create/update GitHub Issues from reports
    
    Args:
        reports: Report list
        github_token: GitHub token
        repo: Repository (org/repo format)
        commit_sha: Commit SHA (optional)
        max_issues: Maximum number of issues
        
    Returns:
        List of created/updated Issues
    """
    manager = GitHubIssueManager(github_token, repo)
    
    created_issues = []
    
    # Process only top-K reports
    top_reports = reports[:max_issues]
    
    for report in top_reports:
        try:
            issue = manager.create_or_update_issue(report, commit_sha)
            created_issues.append(issue)
        except Exception as e:
            print(f"Failed to create/update issue: {e}")
            continue
    
    # Create summary issue if there are more reports
    if len(reports) > max_issues:
        summary_title = f"[NNAST] {len(reports) - max_issues} Additional Findings"
        summary_body = f"## Summary\n\n{len(reports) - max_issues} additional security findings were detected but not individually reported.\n\n"
        summary_body += "Please review the full scan results.\n"
        
        summary_labels = ["security", "nnast", "severity:medium"]
        
        try:
            summary_issue = manager.repo.create_issue(
                title=summary_title,
                body=summary_body,
                labels=summary_labels
            )
            created_issues.append(summary_issue)
        except Exception as e:
            print(f"Failed to create summary issue: {e}")
    
    return created_issues
