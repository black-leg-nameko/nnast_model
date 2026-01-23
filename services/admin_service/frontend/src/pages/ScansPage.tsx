import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { Link, useSearchParams } from 'react-router-dom'
import { scansApi, projectsApi } from '../api/client'

const ScansPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams()
  const [page, setPage] = useState(1)

  const projectId = searchParams.get('project') || undefined
  const status = searchParams.get('status') || undefined

  const { data: scans, isLoading } = useQuery(
    ['scans', page, projectId, status],
    () => scansApi.list(page, 20, projectId, status),
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

  const getStatusClass = (st: string) => {
    const classes: Record<string, string> = {
      completed: 'status-success',
      running: 'status-warning',
      queued: 'status-info',
      failed: 'status-danger',
    }
    return classes[st] || ''
  }

  const getTriggerIcon = (trigger: string) => {
    const icons: Record<string, string> = {
      push: '⬆️',
      pull_request: '🔀',
      manual: '▶️',
      scheduled: '⏰',
    }
    return icons[trigger] || '📋'
  }

  const formatDuration = (ms?: number) => {
    if (!ms) return '-'
    const seconds = Math.floor(ms / 1000)
    if (seconds < 60) return `${seconds}s`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}m ${remainingSeconds}s`
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString('en', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <div className="scans-page">
      <div className="page-header">
        <div>
          <h1 className="page-title">Scans</h1>
          <p className="page-subtitle">
            {scans?.total || 0} security scans across your projects
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
          <label className="filter-label">Status</label>
          <select
            className="filter-select"
            value={status || ''}
            onChange={(e) => updateFilter('status', e.target.value)}
          >
            <option value="">All Status</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="queued">Queued</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      {/* Scans Table */}
      {isLoading ? (
        <div className="loading">Loading scans...</div>
      ) : scans?.items.length === 0 ? (
        <div className="empty-state-card">
          <span className="empty-icon">🔍</span>
          <h3>No scans found</h3>
          <p>Configure a project and push some code to trigger a scan.</p>
        </div>
      ) : (
        <>
          <div className="scans-table-container">
            <table className="scans-table">
              <thead>
                <tr>
                  <th>Repository</th>
                  <th>Branch / Commit</th>
                  <th>Trigger</th>
                  <th>Status</th>
                  <th>Findings</th>
                  <th>Duration</th>
                  <th>Started</th>
                </tr>
              </thead>
              <tbody>
                {scans?.items.map((scan) => (
                  <tr key={scan.scan_id}>
                    <td>
                      <div className="scan-repo">
                        <strong>{scan.repo.split('/')[1]}</strong>
                        <span className="repo-org">{scan.repo.split('/')[0]}</span>
                      </div>
                    </td>
                    <td>
                      <div className="scan-commit">
                        <span className="branch-name">{scan.branch}</span>
                        <code className="commit-sha">{scan.commit_sha.slice(0, 7)}</code>
                        {scan.pr_number && (
                          <span className="pr-info">PR #{scan.pr_number}</span>
                        )}
                      </div>
                      <div className="commit-message">{scan.commit_message}</div>
                    </td>
                    <td>
                      <span className="trigger-badge">
                        {getTriggerIcon(scan.trigger)} {scan.trigger.replace('_', ' ')}
                      </span>
                    </td>
                    <td>
                      <span className={`status-badge ${getStatusClass(scan.status)}`}>
                        {scan.status}
                      </span>
                    </td>
                    <td>
                      {scan.status === 'completed' ? (
                        <div className="findings-summary">
                          <span className="findings-total">{scan.findings_count}</span>
                          {scan.findings_count > 0 && (
                            <div className="findings-breakdown">
                              {scan.critical_count > 0 && (
                                <span className="severity-critical">{scan.critical_count}C</span>
                              )}
                              {scan.high_count > 0 && (
                                <span className="severity-high">{scan.high_count}H</span>
                              )}
                              {scan.medium_count > 0 && (
                                <span className="severity-medium">{scan.medium_count}M</span>
                              )}
                              {scan.low_count > 0 && (
                                <span className="severity-low">{scan.low_count}L</span>
                              )}
                            </div>
                          )}
                          {scan.findings_count > 0 && (
                            <Link
                              to={`/findings?scan=${scan.scan_id}`}
                              className="view-findings-link"
                            >
                              View
                            </Link>
                          )}
                        </div>
                      ) : (
                        <span className="pending">-</span>
                      )}
                    </td>
                    <td>
                      <span className="duration">{formatDuration(scan.duration_ms)}</span>
                    </td>
                    <td>
                      <span className="date">{formatDate(scan.created_at)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {scans && scans.total_pages > 1 && (
            <div className="pagination">
              <button
                className="pagination-btn"
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={page === 1}
              >
                Previous
              </button>
              <span className="pagination-info">
                Page {page} of {scans.total_pages}
              </span>
              <button
                className="pagination-btn"
                onClick={() => setPage(p => Math.min(scans.total_pages, p + 1))}
                disabled={page >= scans.total_pages}
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

export default ScansPage
