import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { Link, useSearchParams } from 'react-router-dom'
import { findingsApi, projectsApi } from '../api/client'
import { OWASP_CATEGORIES } from '../types'

const FindingsPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams()
  const [page, setPage] = useState(1)

  const projectId = searchParams.get('project') || undefined
  const severity = searchParams.get('severity') || undefined
  const status = searchParams.get('status') || undefined
  const owaspCategory = searchParams.get('owasp') || undefined

  const { data: findings, isLoading } = useQuery(
    ['findings', page, projectId, severity, status, owaspCategory],
    () => findingsApi.list(page, 20, { projectId, severity, status, owaspCategory }),
    { keepPreviousData: true }
  )

  const { data: projects } = useQuery('projects', () => projectsApi.list(1, 100))

  const updateFilter = (key: string, value: string) => {
    const newParams = new URLSearchParams(searchParams)
    if (value) {
      newParams.set(key, value)
    } else {
      newParams.delete(key)
    }
    setSearchParams(newParams)
    setPage(1)
  }

  const getSeverityClass = (sev: string) => {
    const classes: Record<string, string> = {
      critical: 'severity-critical',
      high: 'severity-high',
      medium: 'severity-medium',
      low: 'severity-low',
    }
    return classes[sev] || ''
  }

  const getStatusClass = (st: string) => {
    const classes: Record<string, string> = {
      open: 'status-open',
      fixed: 'status-fixed',
      ignored: 'status-ignored',
      false_positive: 'status-false-positive',
    }
    return classes[st] || ''
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <div className="findings-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Findings</h1>
          <p className="page-subtitle">
            {findings?.total || 0} vulnerabilities detected across your projects
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="filters-bar">
        <div className="filter-group">
          <label className="filter-label">Project</label>
          <select
            className="filter-select"
            value={projectId || ''}
            onChange={(e) => updateFilter('project', e.target.value)}
          >
            <option value="">All Projects</option>
            {projects?.items.map((project) => (
              <option key={project.project_id} value={project.project_id}>
                {project.name}
              </option>
            ))}
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">Severity</label>
          <select
            className="filter-select"
            value={severity || ''}
            onChange={(e) => updateFilter('severity', e.target.value)}
          >
            <option value="">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">Status</label>
          <select
            className="filter-select"
            value={status || ''}
            onChange={(e) => updateFilter('status', e.target.value)}
          >
            <option value="">All Status</option>
            <option value="open">Open</option>
            <option value="fixed">Fixed</option>
            <option value="ignored">Ignored</option>
            <option value="false_positive">False Positive</option>
          </select>
        </div>

        <div className="filter-group">
          <label className="filter-label">OWASP Category</label>
          <select
            className="filter-select"
            value={owaspCategory || ''}
            onChange={(e) => updateFilter('owasp', e.target.value)}
          >
            <option value="">All Categories</option>
            {Object.entries(OWASP_CATEGORIES).map(([key, name]) => (
              <option key={key} value={key}>
                {key}: {name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Findings Table */}
      {isLoading ? (
        <div className="loading">Loading findings...</div>
      ) : findings?.items.length === 0 ? (
        <div className="empty-state-card">
          <span className="empty-icon">🎉</span>
          <h3>No findings found</h3>
          <p>Great job! No vulnerabilities match your current filters.</p>
        </div>
      ) : (
        <>
          <div className="findings-table-container">
            <table className="findings-table">
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Finding</th>
                  <th>Location</th>
                  <th>OWASP</th>
                  <th>Status</th>
                  <th>First Seen</th>
                </tr>
              </thead>
              <tbody>
                {findings?.items.map((finding) => (
                  <tr key={finding.finding_id}>
                    <td>
                      <span className={`severity-badge ${getSeverityClass(finding.severity)}`}>
                        {finding.severity.toUpperCase()}
                      </span>
                    </td>
                    <td>
                      <Link to={`/findings/${finding.finding_id}`} className="finding-title-link">
                        <strong>{finding.title}</strong>
                        <span className="finding-rule-id">{finding.rule_id}</span>
                      </Link>
                      <div className="finding-cwe">
                        {finding.cwe_id}: {finding.cwe_name}
                      </div>
                    </td>
                    <td>
                      <code className="finding-location">
                        {finding.location.file}:{finding.location.start_line}
                      </code>
                      {finding.location.function_name && (
                        <div className="finding-function">
                          in {finding.location.function_name}()
                        </div>
                      )}
                    </td>
                    <td>
                      <span className="owasp-badge">
                        {finding.owasp_category}
                      </span>
                      <div className="owasp-name">{finding.owasp_name}</div>
                    </td>
                    <td>
                      <span className={`status-badge ${getStatusClass(finding.status)}`}>
                        {finding.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td>
                      <span className="date">{formatDate(finding.first_seen_at)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {findings && findings.total_pages > 1 && (
            <div className="pagination">
              <button
                className="pagination-btn"
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
              >
                Previous
              </button>
              <span className="pagination-info">
                Page {page} of {findings.total_pages}
              </span>
              <button
                className="pagination-btn"
                onClick={() => setPage(p => Math.min(findings.total_pages, p + 1))}
                disabled={page >= findings.total_pages}
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default FindingsPage
