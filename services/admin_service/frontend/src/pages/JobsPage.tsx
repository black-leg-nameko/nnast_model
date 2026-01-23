import React, { useState } from 'react'
import { useQuery } from 'react-query'
import { jobsApi, Job } from '../api/client'
import { format } from 'date-fns'

const JobsPage: React.FC = () => {
  const [page, setPage] = useState(1)
  const [statusFilter, setStatusFilter] = useState('')

  const { data, isLoading, error } = useQuery(
    ['jobs', page, statusFilter],
    () => jobsApi.list(page, undefined, undefined, statusFilter || undefined).then(res => res.data),
    { keepPreviousData: true }
  )

  const getStatusBadge = (status: string) => {
    const badges: Record<string, string> = {
      queued: 'badge-info',
      running: 'badge-warning',
      done: 'badge-success',
      failed: 'badge-danger',
    }
    return badges[status] || 'badge-secondary'
  }

  if (isLoading) return <div className="loading">Loading...</div>
  if (error) return <div className="error">Error loading jobs</div>

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Jobs</h2>
        <div>
          <select
            className="form-input"
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value)
              setPage(1)
            }}
            style={{ width: '200px', marginRight: '1rem' }}
          >
            <option value="">All Status</option>
            <option value="queued">Queued</option>
            <option value="running">Running</option>
            <option value="done">Done</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Tenant ID</th>
              <th>Project ID</th>
              <th>Repo</th>
              <th>Status</th>
              <th>Findings Count</th>
              <th>Report Count</th>
              <th>Created At</th>
              <th>Finished At</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((job) => (
              <tr key={job.job_id}>
                <td>{job.job_id}</td>
                <td>{job.tenant_id}</td>
                <td>{job.project_id}</td>
                <td>{job.repo}</td>
                <td>
                  <span className={`badge ${getStatusBadge(job.status)}`}>
                    {job.status}
                  </span>
                </td>
                <td>{job.findings_count}</td>
                <td>{job.report_count || '-'}</td>
                <td>{format(new Date(job.created_at), 'yyyy-MM-dd HH:mm:ss')}</td>
                <td>
                  {job.finished_at
                    ? format(new Date(job.finished_at), 'yyyy-MM-dd HH:mm:ss')
                    : '-'}
                </td>
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

export default JobsPage
