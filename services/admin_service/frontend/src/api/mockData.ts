// Mock data for development and demo
import type {
  User,
  Tenant,
  Project,
  Scan,
  Finding,
  DashboardStats,
  AuditLog,
} from '../types'

export const mockUser: User = {
  id: 'user_01',
  email: 'demo@example.com',
  name: 'Demo User',
  avatar_url: 'https://avatars.githubusercontent.com/u/1234567',
  github_login: 'demo-user',
  tenant_id: 'tenant_01',
  role: 'admin',
  created_at: '2024-01-15T09:00:00Z',
}

export const mockTenant: Tenant = {
  tenant_id: 'tenant_01',
  name: 'Demo Organization',
  plan: 'team',
  created_at: '2024-01-15T09:00:00Z',
  updated_at: '2024-01-15T09:00:00Z',
}

export const mockProjects: Project[] = [
  {
    project_id: 'proj_01',
    tenant_id: 'tenant_01',
    name: 'flask-webapp',
    github_org: 'demo-org',
    github_repo: 'flask-webapp',
    github_url: 'https://github.com/demo-org/flask-webapp',
    default_branch: 'main',
    allowed_subject_patterns: ['repo:demo-org/flask-webapp:*'],
    settings: {
      top_k: 5,
      severity_threshold: 'medium',
      auto_create_issues: true,
      scan_on_push: true,
      scan_on_pr: true,
    },
    stats: {
      total_scans: 45,
      total_findings: 23,
      open_findings: 8,
      fixed_findings: 15,
    },
    created_at: '2024-01-20T10:00:00Z',
    updated_at: '2024-06-15T14:30:00Z',
    last_scan_at: '2024-06-15T14:30:00Z',
  },
  {
    project_id: 'proj_02',
    tenant_id: 'tenant_01',
    name: 'django-api',
    github_org: 'demo-org',
    github_repo: 'django-api',
    github_url: 'https://github.com/demo-org/django-api',
    default_branch: 'main',
    allowed_subject_patterns: ['repo:demo-org/django-api:*'],
    settings: {
      top_k: 3,
      severity_threshold: 'high',
      auto_create_issues: true,
      scan_on_push: true,
      scan_on_pr: true,
    },
    stats: {
      total_scans: 128,
      total_findings: 67,
      open_findings: 12,
      fixed_findings: 55,
    },
    created_at: '2024-02-01T09:00:00Z',
    updated_at: '2024-06-14T16:45:00Z',
    last_scan_at: '2024-06-14T16:45:00Z',
  },
  {
    project_id: 'proj_03',
    tenant_id: 'tenant_01',
    name: 'fastapi-service',
    github_org: 'demo-org',
    github_repo: 'fastapi-service',
    github_url: 'https://github.com/demo-org/fastapi-service',
    default_branch: 'main',
    allowed_subject_patterns: ['repo:demo-org/fastapi-service:*'],
    settings: {
      top_k: 5,
      severity_threshold: 'medium',
      auto_create_issues: false,
      scan_on_push: true,
      scan_on_pr: true,
    },
    stats: {
      total_scans: 32,
      total_findings: 11,
      open_findings: 3,
      fixed_findings: 8,
    },
    created_at: '2024-03-10T11:00:00Z',
    updated_at: '2024-06-13T09:20:00Z',
    last_scan_at: '2024-06-13T09:20:00Z',
  },
]

export const mockScans: Scan[] = [
  {
    scan_id: 'scan_001',
    project_id: 'proj_01',
    tenant_id: 'tenant_01',
    repo: 'demo-org/flask-webapp',
    branch: 'main',
    commit_sha: 'a1b2c3d4e5f6',
    commit_message: 'feat: Add user authentication module',
    trigger: 'push',
    status: 'completed',
    findings_count: 3,
    critical_count: 1,
    high_count: 1,
    medium_count: 1,
    low_count: 0,
    duration_ms: 45000,
    created_at: '2024-06-15T14:25:00Z',
    started_at: '2024-06-15T14:25:30Z',
    finished_at: '2024-06-15T14:30:00Z',
  },
  {
    scan_id: 'scan_002',
    project_id: 'proj_02',
    tenant_id: 'tenant_01',
    repo: 'demo-org/django-api',
    branch: 'feature/payments',
    commit_sha: 'b2c3d4e5f6g7',
    commit_message: 'fix: Update payment validation',
    trigger: 'pull_request',
    status: 'completed',
    findings_count: 2,
    critical_count: 0,
    high_count: 2,
    medium_count: 0,
    low_count: 0,
    duration_ms: 62000,
    created_at: '2024-06-14T16:40:00Z',
    started_at: '2024-06-14T16:40:15Z',
    finished_at: '2024-06-14T16:45:00Z',
    pr_number: 142,
    pr_title: 'Update payment validation logic',
  },
  {
    scan_id: 'scan_003',
    project_id: 'proj_01',
    tenant_id: 'tenant_01',
    repo: 'demo-org/flask-webapp',
    branch: 'develop',
    commit_sha: 'c3d4e5f6g7h8',
    commit_message: 'refactor: Clean up database queries',
    trigger: 'push',
    status: 'running',
    findings_count: 0,
    critical_count: 0,
    high_count: 0,
    medium_count: 0,
    low_count: 0,
    created_at: '2024-06-15T15:00:00Z',
    started_at: '2024-06-15T15:00:30Z',
  },
]

export const mockFindings: Finding[] = [
  {
    finding_id: 'find_001',
    scan_id: 'scan_001',
    project_id: 'proj_01',
    tenant_id: 'tenant_01',
    fingerprint: 'fp_sql_injection_001',
    status: 'open',
    severity: 'critical',
    confidence: 0.95,
    rule_id: 'NNAST-INJ-001',
    rule_name: 'SQL Injection via String Formatting',
    cwe_id: 'CWE-89',
    cwe_name: 'SQL Injection',
    owasp_category: 'A03',
    owasp_name: 'Injection',
    title: 'SQL Injection in user query',
    description: 'User input is directly concatenated into SQL query without proper sanitization, allowing potential SQL injection attacks.',
    location: {
      file: 'src/models/user.py',
      start_line: 45,
      end_line: 48,
      function_name: 'get_user_by_email',
      class_name: 'UserRepository',
    },
    code_snippet: {
      code: `def get_user_by_email(self, email: str) -> User:
    query = f"SELECT * FROM users WHERE email = '{email}'"
    result = self.db.execute(query)
    return User.from_row(result.fetchone())`,
      language: 'python',
      start_line: 44,
      highlighted_lines: [45, 46],
    },
    data_flow: [
      {
        step: 1,
        type: 'source',
        location: {
          file: 'src/routes/auth.py',
          start_line: 23,
          end_line: 23,
        },
        code: 'email = request.form.get("email")',
        description: 'User input from HTTP request',
      },
      {
        step: 2,
        type: 'propagation',
        location: {
          file: 'src/routes/auth.py',
          start_line: 25,
          end_line: 25,
        },
        code: 'user = user_repo.get_user_by_email(email)',
        description: 'Passed to repository method',
      },
      {
        step: 3,
        type: 'sink',
        location: {
          file: 'src/models/user.py',
          start_line: 46,
          end_line: 46,
        },
        code: 'result = self.db.execute(query)',
        description: 'Unsanitized input reaches SQL execution',
      },
    ],
    remediation: {
      suggestion: 'Use parameterized queries instead of string formatting to prevent SQL injection.',
      fix_example: `def get_user_by_email(self, email: str) -> User:
    query = "SELECT * FROM users WHERE email = ?"
    result = self.db.execute(query, (email,))
    return User.from_row(result.fetchone())`,
      references: [
        'https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html',
        'https://cwe.mitre.org/data/definitions/89.html',
      ],
      generated_at: '2024-06-15T14:30:00Z',
    },
    created_at: '2024-06-15T14:30:00Z',
    updated_at: '2024-06-15T14:30:00Z',
    first_seen_at: '2024-06-15T14:30:00Z',
  },
  {
    finding_id: 'find_002',
    scan_id: 'scan_001',
    project_id: 'proj_01',
    tenant_id: 'tenant_01',
    fingerprint: 'fp_path_traversal_001',
    status: 'open',
    severity: 'high',
    confidence: 0.88,
    rule_id: 'NNAST-PT-001',
    rule_name: 'Path Traversal in File Operations',
    cwe_id: 'CWE-22',
    cwe_name: 'Path Traversal',
    owasp_category: 'A01',
    owasp_name: 'Broken Access Control',
    title: 'Path Traversal vulnerability in file download',
    description: 'User-controlled filename is used directly in file path construction, allowing potential directory traversal attacks.',
    location: {
      file: 'src/routes/files.py',
      start_line: 67,
      end_line: 70,
      function_name: 'download_file',
    },
    code_snippet: {
      code: `@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    return send_file(file_path)`,
      language: 'python',
      start_line: 66,
      highlighted_lines: [68, 69],
    },
    data_flow: [
      {
        step: 1,
        type: 'source',
        location: {
          file: 'src/routes/files.py',
          start_line: 66,
          end_line: 66,
        },
        code: "@app.route('/download/<filename>')",
        description: 'User input from URL parameter',
      },
      {
        step: 2,
        type: 'sink',
        location: {
          file: 'src/routes/files.py',
          start_line: 68,
          end_line: 68,
        },
        code: 'file_path = os.path.join(UPLOAD_DIR, filename)',
        description: 'Unsanitized input used in file path',
      },
    ],
    remediation: {
      suggestion: 'Validate and sanitize the filename to prevent directory traversal. Use secure_filename() from werkzeug and verify the resolved path is within the allowed directory.',
      fix_example: `from werkzeug.utils import secure_filename

@app.route('/download/<filename>')
def download_file(filename):
    filename = secure_filename(filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    # Verify the resolved path is within UPLOAD_DIR
    if not os.path.realpath(file_path).startswith(os.path.realpath(UPLOAD_DIR)):
        abort(403)
    return send_file(file_path)`,
      references: [
        'https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html',
        'https://cwe.mitre.org/data/definitions/22.html',
      ],
      generated_at: '2024-06-15T14:30:00Z',
    },
    created_at: '2024-06-15T14:30:00Z',
    updated_at: '2024-06-15T14:30:00Z',
    first_seen_at: '2024-06-15T14:30:00Z',
  },
  {
    finding_id: 'find_003',
    scan_id: 'scan_001',
    project_id: 'proj_01',
    tenant_id: 'tenant_01',
    fingerprint: 'fp_xss_001',
    status: 'fixed',
    severity: 'medium',
    confidence: 0.82,
    rule_id: 'NNAST-XSS-001',
    rule_name: 'Reflected XSS via Template',
    cwe_id: 'CWE-79',
    cwe_name: 'Cross-site Scripting (XSS)',
    owasp_category: 'A03',
    owasp_name: 'Injection',
    title: 'Potential XSS in error message display',
    description: 'User input is reflected in the response without proper encoding, potentially allowing XSS attacks.',
    location: {
      file: 'src/routes/search.py',
      start_line: 32,
      end_line: 34,
      function_name: 'search',
    },
    code_snippet: {
      code: `@app.route('/search')
def search():
    query = request.args.get('q', '')
    return render_template('search.html', query=query)`,
      language: 'python',
      start_line: 31,
      highlighted_lines: [33, 34],
    },
    remediation: {
      suggestion: 'Ensure the template properly escapes the query variable. Jinja2 auto-escapes by default, but verify the template does not use |safe filter.',
      references: [
        'https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html',
      ],
      generated_at: '2024-06-15T14:30:00Z',
    },
    created_at: '2024-06-01T10:00:00Z',
    updated_at: '2024-06-10T11:30:00Z',
    first_seen_at: '2024-06-01T10:00:00Z',
    fixed_at: '2024-06-10T11:30:00Z',
  },
  {
    finding_id: 'find_004',
    scan_id: 'scan_002',
    project_id: 'proj_02',
    tenant_id: 'tenant_01',
    fingerprint: 'fp_ssrf_001',
    status: 'open',
    severity: 'high',
    confidence: 0.91,
    rule_id: 'NNAST-SSRF-001',
    rule_name: 'Server-Side Request Forgery',
    cwe_id: 'CWE-918',
    cwe_name: 'Server-Side Request Forgery (SSRF)',
    owasp_category: 'A10',
    owasp_name: 'Server-Side Request Forgery',
    title: 'SSRF vulnerability in URL fetch',
    description: 'User-provided URL is fetched without validation, allowing potential SSRF attacks to access internal services.',
    location: {
      file: 'src/services/webhook.py',
      start_line: 45,
      end_line: 48,
      function_name: 'fetch_webhook_data',
    },
    code_snippet: {
      code: `def fetch_webhook_data(url: str) -> dict:
    response = requests.get(url)
    return response.json()`,
      language: 'python',
      start_line: 44,
      highlighted_lines: [45, 46],
    },
    remediation: {
      suggestion: 'Validate and sanitize the URL before making requests. Block internal IP ranges and use an allowlist of permitted domains.',
      fix_example: `import ipaddress
from urllib.parse import urlparse

ALLOWED_DOMAINS = ['api.example.com', 'webhook.example.com']

def is_safe_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_DOMAINS:
        return False
    try:
        ip = ipaddress.ip_address(parsed.hostname)
        if ip.is_private or ip.is_loopback:
            return False
    except ValueError:
        pass
    return True

def fetch_webhook_data(url: str) -> dict:
    if not is_safe_url(url):
        raise ValueError("URL not allowed")
    response = requests.get(url, timeout=10)
    return response.json()`,
      references: [
        'https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html',
      ],
      generated_at: '2024-06-14T16:45:00Z',
    },
    created_at: '2024-06-14T16:45:00Z',
    updated_at: '2024-06-14T16:45:00Z',
    first_seen_at: '2024-06-14T16:45:00Z',
  },
]

export const mockDashboardStats: DashboardStats = {
  projects: {
    total: 3,
    active: 3,
  },
  scans: {
    total: 205,
    this_week: 18,
    this_month: 67,
  },
  findings: {
    total: 101,
    open: 23,
    fixed: 78,
    by_severity: {
      critical: 5,
      high: 12,
      medium: 4,
      low: 2,
    },
    by_owasp: {
      'A01': 8,
      'A02': 3,
      'A03': 15,
      'A04': 2,
      'A05': 4,
      'A06': 1,
      'A07': 5,
      'A08': 2,
      'A09': 1,
      'A10': 3,
    },
  },
  trend: [
    { date: '2024-06-09', scans: 8, findings: 12, fixed: 5 },
    { date: '2024-06-10', scans: 12, findings: 8, fixed: 10 },
    { date: '2024-06-11', scans: 6, findings: 4, fixed: 3 },
    { date: '2024-06-12', scans: 15, findings: 18, fixed: 8 },
    { date: '2024-06-13', scans: 9, findings: 6, fixed: 12 },
    { date: '2024-06-14', scans: 11, findings: 9, fixed: 7 },
    { date: '2024-06-15', scans: 7, findings: 5, fixed: 4 },
  ],
}

export const mockAuditLogs: AuditLog[] = [
  {
    log_id: 'log_001',
    tenant_id: 'tenant_01',
    actor: 'demo-user',
    actor_type: 'user',
    action: 'project.create',
    resource_type: 'project',
    resource_id: 'proj_03',
    result: 'success',
    timestamp: '2024-06-15T14:00:00Z',
  },
  {
    log_id: 'log_002',
    tenant_id: 'tenant_01',
    actor: 'github-actions',
    actor_type: 'ci',
    action: 'scan.complete',
    resource_type: 'scan',
    resource_id: 'scan_001',
    result: 'success',
    timestamp: '2024-06-15T14:30:00Z',
  },
  {
    log_id: 'log_003',
    tenant_id: 'tenant_01',
    actor: 'demo-user',
    actor_type: 'user',
    action: 'finding.ignore',
    resource_type: 'finding',
    resource_id: 'find_003',
    result: 'success',
    metadata: { reason: 'False positive - sanitization exists in template' },
    timestamp: '2024-06-15T15:10:00Z',
  },
]
