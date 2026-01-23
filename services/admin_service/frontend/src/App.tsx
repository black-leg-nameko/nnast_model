import React from 'react'
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom'
import TenantsPage from './pages/TenantsPage'
import ProjectsPage from './pages/ProjectsPage'
import JobsPage from './pages/JobsPage'
import AuditLogsPage from './pages/AuditLogsPage'
import './App.css'

function App() {
  return (
    <Router>
      <div className="app">
        <nav className="navbar">
          <div className="navbar-brand">
            <h1>NNAST Admin</h1>
          </div>
          <div className="navbar-menu">
            <Link to="/tenants" className="nav-link">Tenants</Link>
            <Link to="/projects" className="nav-link">Projects</Link>
            <Link to="/jobs" className="nav-link">Jobs</Link>
            <Link to="/audit-logs" className="nav-link">Audit Logs</Link>
          </div>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/" element={<TenantsPage />} />
            <Route path="/tenants" element={<TenantsPage />} />
            <Route path="/projects" element={<ProjectsPage />} />
            <Route path="/jobs" element={<JobsPage />} />
            <Route path="/audit-logs" element={<AuditLogsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
