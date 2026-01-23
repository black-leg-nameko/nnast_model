import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { projectsApi, Project } from '../api/client'
import { format } from 'date-fns'

const ProjectsPage: React.FC = () => {
  const [page, setPage] = useState(1)
  const [showModal, setShowModal] = useState(false)
  const [editingProject, setEditingProject] = useState<Project | null>(null)
  const [formData, setFormData] = useState({
    project_id: '',
    tenant_id: '',
    github_org: '',
    github_repo: '',
    allowed_subject_patterns: [] as string[],
  })
  const queryClient = useQueryClient()

  const { data, isLoading, error } = useQuery(
    ['projects', page],
    () => projectsApi.list(page).then(res => res.data),
    { keepPreviousData: true }
  )

  const createMutation = useMutation(
    (project: Omit<Project, 'created_at' | 'updated_at'>) => projectsApi.create(project),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('projects')
        setShowModal(false)
        resetForm()
      },
    }
  )

  const updateMutation = useMutation(
    ({ projectId, updates }: { projectId: string; updates: Partial<Project> }) =>
      projectsApi.update(projectId, updates),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('projects')
        setShowModal(false)
        setEditingProject(null)
        resetForm()
      },
    }
  )

  const deleteMutation = useMutation(
    (projectId: string) => projectsApi.delete(projectId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('projects')
      },
    }
  )

  const resetForm = () => {
    setFormData({
      project_id: '',
      tenant_id: '',
      github_org: '',
      github_repo: '',
      allowed_subject_patterns: [],
    })
  }

  const handleEdit = (project: Project) => {
    setEditingProject(project)
    setFormData({
      project_id: project.project_id,
      tenant_id: project.tenant_id,
      github_org: project.github_org,
      github_repo: project.github_repo,
      allowed_subject_patterns: project.allowed_subject_patterns || [],
    })
    setShowModal(true)
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (editingProject) {
      updateMutation.mutate({
        projectId: editingProject.project_id,
        updates: formData,
      })
    } else {
      createMutation.mutate(formData)
    }
  }

  const getStatusBadge = (status: string) => {
    const badges: Record<string, string> = {
      active: 'badge-success',
      inactive: 'badge-secondary',
    }
    return badges[status] || 'badge-info'
  }

  if (isLoading) return <div className="loading">Loading...</div>
  if (error) return <div className="error">Error loading projects</div>

  return (
    <div>
      <div className="page-header">
        <h2 className="page-title">Projects</h2>
        <button className="btn btn-primary" onClick={() => {
          setEditingProject(null)
          resetForm()
          setShowModal(true)
        }}>
          Create Project
        </button>
      </div>

      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Project ID</th>
              <th>Tenant ID</th>
              <th>GitHub Org</th>
              <th>GitHub Repo</th>
              <th>Created At</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {data?.items.map((project) => (
              <tr key={project.project_id}>
                <td>{project.project_id}</td>
                <td>{project.tenant_id}</td>
                <td>{project.github_org}</td>
                <td>{project.github_repo}</td>
                <td>{format(new Date(project.created_at), 'yyyy-MM-dd HH:mm:ss')}</td>
                <td>
                  <button className="btn btn-secondary" onClick={() => handleEdit(project)}>
                    Edit
                  </button>
                  <button
                    className="btn btn-danger"
                    onClick={() => {
                      if (window.confirm('Are you sure you want to delete this project?')) {
                        deleteMutation.mutate(project.project_id)
                      }
                    }}
                  >
                    Delete
                  </button>
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

      {showModal && (
        <div className="modal" onClick={() => setShowModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">
                {editingProject ? 'Edit Project' : 'Create Project'}
              </h3>
              <button className="close-btn" onClick={() => setShowModal(false)}>×</button>
            </div>
            <form onSubmit={handleSubmit}>
              {!editingProject && (
                <div className="form-group">
                  <label className="form-label">Project ID</label>
                  <input
                    type="text"
                    className="form-input"
                    value={formData.project_id}
                    onChange={(e) => setFormData({ ...formData, project_id: e.target.value })}
                    required
                  />
                </div>
              )}
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
                <label className="form-label">GitHub Org</label>
                <input
                  type="text"
                  className="form-input"
                  value={formData.github_org}
                  onChange={(e) => setFormData({ ...formData, github_org: e.target.value })}
                  required
                />
              </div>
              <div className="form-group">
                <label className="form-label">GitHub Repo</label>
                <input
                  type="text"
                  className="form-input"
                  value={formData.github_repo}
                  onChange={(e) => setFormData({ ...formData, github_repo: e.target.value })}
                  required
                />
              </div>
              <div className="form-actions">
                <button type="button" className="btn btn-secondary" onClick={() => setShowModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  {editingProject ? 'Update' : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}

export default ProjectsPage
