// Core types for NNAST Dashboard

// User & Auth
export interface User {
  id: string
  email: string
  name: string
  avatar_url: string
  github_login: string
  tenant_id: string
  role: 'admin' | 'member' | 'viewer'
  created_at: string
}

export interface AuthState {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
}

// Tenant & Project
export interface Tenant {
  tenant_id: string
  name: string
  plan: 'free' | 'team' | 'enterprise'
  created_at: string
  updated_at: string
}

export interface Project {
  project_id: string
  tenant_id: string
  name: string
  github_org: string
  github_repo: string
  github_url: string
  default_branch: string
  allowed_subject_patterns: string[]
  settings: ProjectSettings
  stats: ProjectStats
  created_at: string
  updated_at: string
  last_scan_at?: string
}

export interface ProjectSettings {
  top_k: number
  severity_threshold: 'critical' | 'high' | 'medium' | 'low'
  auto_create_issues: boolean
  scan_on_push: boolean
  scan_on_pr: boolean
}

export interface ProjectStats {
  total_scans: number
  total_findings: number
  open_findings: number
  fixed_findings: number
}

// Scan
export interface Scan {
  scan_id: string
  project_id: string
  tenant_id: string
  repo: string
  branch: string
  commit_sha: string
  commit_message: string
  trigger: 'push' | 'pull_request' | 'manual' | 'scheduled'
  status: 'queued' | 'running' | 'completed' | 'failed'
  findings_count: number
  critical_count: number
  high_count: number
  medium_count: number
  low_count: number
  duration_ms?: number
  created_at: string
  started_at?: string
  finished_at?: string
  pr_number?: number
  pr_title?: string
}

// Finding (Vulnerability)
export interface Finding {
  finding_id: string
  scan_id: string
  project_id: string
  tenant_id: string
  fingerprint: string
  status: 'open' | 'fixed' | 'ignored' | 'false_positive'
  severity: 'critical' | 'high' | 'medium' | 'low'
  confidence: number
  rule_id: string
  rule_name: string
  cwe_id: string
  cwe_name: string
  owasp_category: string
  owasp_name: string
  title: string
  description: string
  location: FindingLocation
  code_snippet: CodeSnippet
  data_flow?: DataFlowPath[]
  remediation?: Remediation
  created_at: string
  updated_at: string
  first_seen_at: string
  fixed_at?: string
  ignored_at?: string
  ignored_by?: string
  ignore_reason?: string
}

export interface FindingLocation {
  file: string
  start_line: number
  end_line: number
  start_column?: number
  end_column?: number
  function_name?: string
  class_name?: string
}

export interface CodeSnippet {
  code: string
  language: string
  start_line: number
  highlighted_lines: number[]
}

export interface DataFlowPath {
  step: number
  type: 'source' | 'propagation' | 'sink'
  location: FindingLocation
  code: string
  description: string
}

export interface Remediation {
  suggestion: string
  fix_example?: string
  references: string[]
  generated_at: string
}

// OWASP Categories
export const OWASP_CATEGORIES = {
  'A01': 'Broken Access Control',
  'A02': 'Cryptographic Failures',
  'A03': 'Injection',
  'A04': 'Insecure Design',
  'A05': 'Security Misconfiguration',
  'A06': 'Vulnerable and Outdated Components',
  'A07': 'Identification and Authentication Failures',
  'A08': 'Software and Data Integrity Failures',
  'A09': 'Security Logging and Monitoring Failures',
  'A10': 'Server-Side Request Forgery',
} as const

export type OWASPCategory = keyof typeof OWASP_CATEGORIES

// Dashboard Stats
export interface DashboardStats {
  projects: {
    total: number
    active: number
  }
  scans: {
    total: number
    this_week: number
    this_month: number
  }
  findings: {
    total: number
    open: number
    fixed: number
    by_severity: {
      critical: number
      high: number
      medium: number
      low: number
    }
    by_owasp: Record<string, number>
  }
  trend: TrendData[]
}

export interface TrendData {
  date: string
  scans: number
  findings: number
  fixed: number
}

// Audit Log
export interface AuditLog {
  log_id: string
  tenant_id: string
  actor: string
  actor_type: 'user' | 'system' | 'ci'
  action: string
  resource_type: string
  resource_id: string
  result: 'success' | 'failure'
  ip_address?: string
  user_agent?: string
  metadata?: Record<string, unknown>
  timestamp: string
}

// API Response Types
export interface PaginatedResponse<T> {
  items: T[]
  page: number
  page_size: number
  total: number
  total_pages: number
}

export interface ApiError {
  code: string
  message: string
  details?: Record<string, unknown>
}
