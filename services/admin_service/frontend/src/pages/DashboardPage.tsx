import React from 'react'
import { useQuery } from 'react-query'
import { Link } from 'react-router-dom'
import { dashboardApi, scansApi, findingsApi } from '../api/client'
import { OWASP_CATEGORIES } from '../types'

const DashboardPage: React.FC = () => {
  const { data: stats, isLoading: statsLoading } = useQuery(
    'dashboardStats',
    () => dashboardApi.getStats(),
    { staleTime: 60000 }
  )

  const { data: recentScans } = useQuery(
    ['scans', 'recent'],
    () => scansApi.list(1, 5),
    { staleTime: 30000 }
  )

  const { data: openFindings } = useQuery(
    ['findings', 'open'],
    () => findingsApi.list(1, 5, { status: 'open' }),
    { staleTime: 30000 }
  )

  if (statsLoading) {
    return <div className="loading">Loading dashboard...</div>
  }

  const getSeverityClass = (severity: string) => {
    const classes: Record<string, string> = {
      critical: 'severity-critical',
      high: 'severity-high',
      medium: 'severity-medium',
      low: 'severity-low',
    }
    return classes[severity] || ''
  }

  const getStatusClass = (status: string) => {
    const classes: Record<string, string> = {
      completed: 'status-success',
      running: 'status-warning',
      queued: 'status-info',
      failed: 'status-danger',
    }
    return classes[status] || ''
  }

  return (
    <div className="dashboard-page">
      <div className="page-header">
        <h1 className="page-title">Security Dashboard</h1>
        <p className="page-subtitle">Overview of your application security posture</p>
      </div>

      {/* Stats Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">📁</div>
          <div className="stat-content">
            <span className="stat-value">{stats?.projects.total || 0}</span>
            <span className="stat-label">Projects</span>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">🔍</div>
          <div className="stat-content">
            <span className="stat-value">{stats?.scans.this_week || 0}</span>
            <span className="stat-label">Scans This Week</span>
          </div>
        </div>

        <div className="stat-card stat-card-warning">
          <div className="stat-icon">⚠️</div>
          <div className="stat-content">
            <span className="stat-value">{stats?.findings.open || 0}</span>
            <span className="stat-label">Open Findings</span>
          </div>
        </div>

        <div className="stat-card stat-card-success">
          <div className="stat-icon">✅</div>
          <div className="stat-content">
            <span className="stat-value">{stats?.findings.fixed || 0}</span>
            <span className="stat-label">Fixed</span>
          </div>
        </div>
      </div>

      {/* Severity Breakdown */}
      <div className="dashboard-section">
        <h2 className="section-title">Open Findings by Severity</h2>
        <div className="severity-grid">
          <div className="severity-card severity-critical">
            <span className="severity-count">{stats?.findings.by_severity.critical || 0}</span>
            <span className="severity-label">Critical</span>
          </div>
          <div className="severity-card severity-high">
            <span className="severity-count">{stats?.findings.by_severity.high || 0}</span>
            <span className="severity-label">High</span>
          </div>
          <div className="severity-card severity-medium">
            <span className="severity-count">{stats?.findings.by_severity.medium || 0}</span>
            <span className="severity-label">Medium</span>
          </div>
          <div className="severity-card severity-low">
            <span className="severity-count">{stats?.findings.by_severity.low || 0}</span>
            <span className="severity-label">Low</span>
          </div>
        </div>
      </div>

      <div className="dashboard-grid">
        {/* Recent Scans */}
        <div className="dashboard-card">
          <div className="card-header">
            <h2 className="card-title">Recent Scans</h2>
            <Link to="/scans" className="card-link">View all</Link>
          </div>
          <div className="card-content">
            {recentScans?.items.length === 0 ? (
              <p className="empty-state">No scans yet</p>
            ) : (
              <ul className="scan-list">
                {recentScans?.items.map((scan) => (
                  <li key={scan.scan_id} className="scan-item">
                    <div className="scan-info">
                      <span className="scan-repo">{scan.repo.split('/')[1]}</span>
                      <span className="scan-branch">{scan.branch}</span>
                    </div>
                    <div className="scan-meta">
                      <span className={`status-badge ${getStatusClass(scan.status)}`}>
                        {scan.status}
                      </span>
                      {scan.findings_count > 0 && (
                        <span className="findings-count">
                          {scan.findings_count} findings
                        </span>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Open Findings */}
        <div className="dashboard-card">
          <div className="card-header">
            <h2 className="card-title">Latest Open Findings</h2>
            <Link to="/findings?status=open" className="card-link">View all</Link>
          </div>
          <div className="card-content">
            {openFindings?.items.length === 0 ? (
              <p className="empty-state">No open findings</p>
            ) : (
              <ul className="finding-list">
                {openFindings?.items.map((finding) => (
                  <li key={finding.finding_id} className="finding-item">
                    <Link to={`/findings/${finding.finding_id}`} className="finding-link">
                      <div className="finding-header">
                        <span className={`severity-badge ${getSeverityClass(finding.severity)}`}>
                          {finding.severity.toUpperCase()}
                        </span>
                        <span className="finding-title">{finding.title}</span>
                      </div>
                      <div className="finding-meta">
                        <span className="finding-location">
                          {finding.location.file}:{finding.location.start_line}
                        </span>
                        <span className="finding-rule">{finding.cwe_id}</span>
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      {/* OWASP Distribution */}
      <div className="dashboard-section">
        <h2 className="section-title">OWASP Top 10 Distribution</h2>
        <div className="owasp-grid">
          {Object.entries(OWASP_CATEGORIES).map(([key, name]) => {
            const count = stats?.findings.by_owasp[key] || 0
            return (
              <Link
                key={key}
                to={`/findings?owasp=${key}`}
                className={`owasp-card ${count > 0 ? 'has-findings' : ''}`}
              >
                <span className="owasp-id">{key}</span>
                <span className="owasp-name">{name}</span>
                <span className="owasp-count">{count}</span>
              </Link>
            )
          })}
        </div>
      </div>

      {/* Trend Chart Placeholder */}
      <div className="dashboard-section">
        <h2 className="section-title">7-Day Trend</h2>
        <div className="trend-chart">
          <div className="trend-bars">
            {stats?.trend.map((day, index) => (
              <div key={index} className="trend-day">
                <div className="trend-bar-container">
                  <div
                    className="trend-bar trend-bar-findings"
                    style={{ height: `${Math.min(day.findings * 5, 100)}%` }}
                    title={`${day.findings} findings`}
                  />
                  <div
                    className="trend-bar trend-bar-fixed"
                    style={{ height: `${Math.min(day.fixed * 5, 100)}%` }}
                    title={`${day.fixed} fixed`}
                  />
                </div>
                <span className="trend-label">
                  {new Date(day.date).toLocaleDateString('en', { weekday: 'short' })}
                </span>
              </div>
            ))}
          </div>
          <div className="trend-legend">
            <span className="legend-item">
              <span className="legend-color findings" />
              New Findings
            </span>
            <span className="legend-item">
              <span className="legend-color fixed" />
              Fixed
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DashboardPage
