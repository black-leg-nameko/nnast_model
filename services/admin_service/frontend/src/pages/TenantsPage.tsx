import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { tenantsApi, Tenant } from '../api/client'
import { format } from 'date-fns'

const TenantsPage: React.FC = () => {
  const [page, setPage] = useState(1)
  const [showModal, setShowModal] = useState(false)
  const [formData, setFormData] = useState({ tenant_id: '', name: '' })
  const queryClient = useQueryClient()

  const { data, isLoading, error } = useQuery(
    ['tenants', page],
    () => tenantsApi.list(page).then(res => res.data),
    { keepPreviousData: true }
  )

  const createMutation = useMutation(
    (tenant: Omit<Tenant, 'created_at' | 'updated_at'>) => tenantsApi.create(tenant),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('tenants')
        setShowModal(false)
        setFormData({ tenant_id: '', name: '' })
      },
    }
  )

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    createMutation.mutate(formData)
  }

  if (isLoading) return <div className="loading">Loading...</div>
  if (error) return <div className="error">Error loading tenants</div>

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Tenants</h2>
        <button className="btn btn-primary" onClick={() => setShowModal(true)}>
          Create Tenant
        </button>
      </div>

      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Tenant ID</th>
              <th>Name</th>
              <th>Created At</th>
              <th>Updated At</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((tenant) => (
              <tr key={tenant.tenant_id}>
                <td>{tenant.tenant_id}</td>
                <td>{tenant.name}</td>
                <td>{format(new Date(tenant.created_at), 'yyyy-MM-dd HH:mm:ss')}</td>
                <td>{format(new Date(tenant.updated_at), 'yyyy-MM-dd HH:mm:ss')}</td>
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

      {showModal && (
        <div className="modal" onClick={() => setShowModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Create Tenant</h3>
              <button className="close-btn" onClick={() => setShowModal(false)}>×</button>
            </div>
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label className="form-label">Tenant ID</label>
                <input
                  type="text"
                  className="form-input"
                  value={formData.tenant_id}
                  onChange={(e) => setFormData({ ...formData, tenant_id: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">Name</label>
                <input
                  type="text"
                  className="form-input"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                />
              </div>
              <div className="form-actions">
                <button type="button" className="btn btn-secondary" onClick={() => setShowModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Create
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

export default TenantsPage
