import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add JWT token to requests
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('jwt_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export interface PaginatedResponse<T> {
  items: T[]
  page: number
  page_size: number
  total: number
  total_pages: number
}

export interface Tenant {
  tenant_id: string
  name: string
  created_at: string
  updated_at: string
}

export interface Project {
  project_id: string
  tenant_id: string
  github_org: string
  github_repo: string
  allowed_subject_patterns: string[]
  created_at: string
  updated_at: string
}

export interface Job {
  job_id: string
  tenant_id: string
  project_id: string
  repo: string
  status: string
  created_at: string
  started_at?: string
  finished_at?: string
  findings_count: number
  report_count?: number
}

export interface AuditLog {
  log_id: string
  tenant_id: string
  actor: string
  action: string
  result: string
  timestamp: string
  metadata?: Record<string, any>
}

export const tenantsApi = {
  list: (page = 1, pageSize = 20) =>
    apiClient.get<PaginatedResponse<Tenant>>('/tenants', {
      params: { page, page_size: pageSize },
    }),
  get: (tenantId: string) =>
    apiClient.get<Tenant>(`/tenants/${tenantId}`),
  create: (tenant: Omit<Tenant, 'created_at' | 'updated_at'>) =>
    apiClient.post<Tenant>('/tenants', tenant),
}

export const projectsApi = {
  list: (page = 1, pageSize = 20, tenantId?: string) =>
    apiClient.get<PaginatedResponse<Project>>('/projects', {
      params: { page, page_size: pageSize, tenant_id: tenantId },
    }),
  get: (projectId: string) =>
    apiClient.get<Project>(`/projects/${projectId}`),
  create: (project: Omit<Project, 'created_at' | 'updated_at'>) =>
    apiClient.post<Project>('/projects', project),
  update: (projectId: string, updates: Partial<Project>) =>
    apiClient.put<Project>(`/projects/${projectId}`, updates),
  delete: (projectId: string) =>
    apiClient.delete(`/projects/${projectId}`),
}

export const jobsApi = {
  list: (page = 1, pageSize = 20, tenantId?: string, projectId?: string, status?: string) =>
    apiClient.get<PaginatedResponse<Job>>('/jobs', {
      params: { page, page_size: pageSize, tenant_id: tenantId, project_id: projectId, status },
    }),
  get: (jobId: string) =>
    apiClient.get<Job>(`/jobs/${jobId}`),
}

export const auditLogsApi = {
  list: (page = 1, pageSize = 20, tenantId?: string, actor?: string) =>
    apiClient.get<PaginatedResponse<AuditLog>>('/audit-logs', {
      params: { page, page_size: pageSize, tenant_id: tenantId, actor },
    }),
}
