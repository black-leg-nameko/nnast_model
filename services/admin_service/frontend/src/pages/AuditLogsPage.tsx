import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { auditLogsApi, AuditLog } from '../api/client'
import { format } from 'date-fns'

const AuditLogsPage: React.FC = () => {
  const [page, setPage] = useState(1)
  const [tenantFilter, setTenantFilter] = useState('')

  const { data, isLoading, error } = useQuery(
    ['audit-logs', page, tenantFilter],
    () => auditLogsApi.list(page, undefined, tenantFilter || undefined).then(res => res.data),
    { keepPreviousData: true }
  )

  const getResultBadge = (result: string) => {
    const badges: Record<string, string> = {
      success: 'badge-success',
      denied: 'badge-danger',
      failed: 'badge-danger',
    }
    return badges[result] || 'badge-info'
  }

  if (isLoading) return <div className="loading">Loading...</div>
  if (error) return <div className="error">Error loading audit logs</div>

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Audit Logs</h2>
        <div>
          <input
            type="text"
            className="form-input"
            placeholder="Filter by Tenant ID"
            value={tenantFilter}
            onChange={(e) => {
              setTenantFilter(e.target.value)
              setPage(1)
            }}
            style={{ width: '200px' }}
          />
        </div>
      </div>

      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Log ID</th>
              <th>Tenant ID</th>
              <th>Actor</th>
              <th>Action</th>
              <th>Result</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((log) => (
              <tr key={log.log_id}>
                <td>{log.log_id}</td>
                <td>{log.tenant_id}</td>
                <td>{log.actor}</td>
                <td>{log.action}</td>
                <td>
                  <span className={`badge ${getResultBadge(log.result)}`}>
                    {log.result}
                  </span>
                </td>
                <td>{format(new Date(log.timestamp), 'yyyy-MM-dd HH:mm:ss')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data && (
        <div className="pagination">
          <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>
            Previous
          </button>
          <span>Page {page} of {data.total_pages}</span>
          <button
            onClick={() => setPage(p => Math.min(data.total_pages, p + 1))}
            disabled={page >= data.total_pages}
          >
            Next
          </button>
        </div>
      )}
    </div>
  )
}

export default AuditLogsPage
