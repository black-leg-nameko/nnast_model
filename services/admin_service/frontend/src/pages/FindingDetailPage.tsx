import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { useParams, Link } from 'react-router-dom'
import { findingsApi } from '../api/client'
import { OWASP_CATEGORIES } from '../types'
import type { Finding } from '../types'

const FindingDetailPage: React.FC = () => {
  const { findingId } = useParams<{ findingId: string }>()
  const queryClient = useQueryClient()
  const [showIgnoreModal, setShowIgnoreModal] = useState(false)
  const [ignoreReason, setIgnoreReason] = useState('')

  const { data: finding, isLoading, error } = useQuery(
    ['finding', findingId],
    () => findingsApi.get(findingId!),
    { enabled: !!findingId }
  )

  const updateStatusMutation = useMutation(
    ({ status, reason }: { status: Finding['status']; reason?: string }) =>
      findingsApi.updateStatus(findingId!, status, reason),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['finding', findingId])
        queryClient.invalidateQueries('findings')
        setShowIgnoreModal(false)
        setIgnoreReason('')
      },
    }
  )

  const getSeverityClass = (severity: string) => {
    const classes: Record<string, string> = {
      critical: 'severity-critical',
      high: 'severity-high',
      medium: 'severity-medium',
      low: 'severity-low',
    }
    return classes[severity] || ''
  }

  if (isLoading) {
    return <div className="loading">Loading finding details...</div>
  }

  if (error || !finding) {
    return (
      <div className="error-state">
        <h2>Finding not found</h2>
        <Link to="/findings" className="btn btn-primary">Back to Findings</Link>
      </div>
    )
  }

  return (
    <div className="finding-detail-page">
      {/* Header */}
      <div className="finding-detail-header">
        <Link to="/findings" className="back-link">← Back to Findings</Link>
        
        <div className="finding-title-row">
          <span className={`severity-badge large ${getSeverityClass(finding.severity)}`}>
            {finding.severity.toUpperCase()}
          </span>
          <h1>{finding.title}</h1>
        </div>

        <div className="finding-meta-row">
          <span className="meta-item">
            <strong>Rule:</strong> {finding.rule_id}
          </span>
          <span className="meta-item">
            <strong>CWE:</strong> {finding.cwe_id} - {finding.cwe_name}
          </span>
          <span className="meta-item">
            <strong>OWASP:</strong> {finding.owasp_category} - {OWASP_CATEGORIES[finding.owasp_category as keyof typeof OWASP_CATEGORIES]}
          </span>
          <span className="meta-item">
            <strong>Confidence:</strong> {Math.round(finding.confidence * 100)}%
          </span>
        </div>

        {/* Status Actions */}
        <div className="finding-actions">
          {finding.status === 'open' && (
            <>
              <button
                className="btn btn-success"
                onClick={() => updateStatusMutation.mutate({ status: 'fixed' })}
                disabled={updateStatusMutation.isLoading}
              >
                ✓ Mark as Fixed
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => setShowIgnoreModal(true)}
              >
                Ignore
              </button>
              <button
                className="btn btn-outline"
                onClick={() => updateStatusMutation.mutate({ status: 'false_positive' })}
                disabled={updateStatusMutation.isLoading}
              >
                False Positive
              </button>
            </>
          )}
          {finding.status !== 'open' && (
            <button
              className="btn btn-outline"
              onClick={() => updateStatusMutation.mutate({ status: 'open' })}
              disabled={updateStatusMutation.isLoading}
            >
              Reopen
            </button>
          )}
          <span className={`status-badge ${finding.status}`}>
            {finding.status.replace('_', ' ')}
          </span>
        </div>
      </div>

      {/* Description */}
      <section className="finding-section">
        <h2>Description</h2>
        <p className="finding-description">{finding.description}</p>
      </section>

      {/* Location */}
      <section className="finding-section">
        <h2>Location</h2>
        <div className="location-info">
          <code className="file-path">{finding.location.file}</code>
          <span className="line-info">
            Lines {finding.location.start_line}-{finding.location.end_line}
          </span>
          {finding.location.function_name && (
            <span className="function-info">
              in <code>{finding.location.class_name ? `${finding.location.class_name}.` : ''}{finding.location.function_name}()</code>
            </span>
          )}
        </div>
      </section>

      {/* Vulnerable Code */}
      <section className="finding-section">
        <h2>Vulnerable Code</h2>
        <div className="code-block">
          <div className="code-header">
            <span className="code-language">{finding.code_snippet.language}</span>
            <span className="code-lines">
              Starting at line {finding.code_snippet.start_line}
            </span>
          </div>
          <pre className="code-content">
            {finding.code_snippet.code.split('\n').map((line, index) => {
              const lineNum = finding.code_snippet.start_line + index
              const isHighlighted = finding.code_snippet.highlighted_lines.includes(lineNum)
              return (
                <div
                  key={index}
                  className={`code-line ${isHighlighted ? 'highlighted' : ''}`}
                >
                  <span className="line-number">{lineNum}</span>
                  <code>{line}</code>
                </div>
              )
            })}
          </pre>
        </div>
      </section>

      {/* Data Flow */}
      {finding.data_flow && finding.data_flow.length > 0 && (
        <section className="finding-section">
          <h2>Data Flow</h2>
          <p className="section-description">
            The path from user input (source) to the vulnerable operation (sink):
          </p>
          <div className="data-flow">
            {finding.data_flow.map((step, index) => (
              <div key={index} className={`flow-step flow-step-${step.type}`}>
                <div className="flow-step-header">
                  <span className="step-number">{step.step}</span>
                  <span className={`step-type ${step.type}`}>{step.type}</span>
                  <span className="step-location">
                    {step.location.file}:{step.location.start_line}
                  </span>
                </div>
                <pre className="step-code"><code>{step.code}</code></pre>
                <p className="step-description">{step.description}</p>
                {index < finding.data_flow!.length - 1 && (
                  <div className="flow-arrow">↓</div>
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Remediation */}
      {finding.remediation && (
        <section className="finding-section">
          <h2>Remediation</h2>
          <div className="remediation-content">
            <p className="remediation-suggestion">{finding.remediation.suggestion}</p>
            
            {finding.remediation.fix_example && (
              <div className="fix-example">
                <h3>Suggested Fix</h3>
                <div className="code-block fix-code">
                  <pre className="code-content">
                    <code>{finding.remediation.fix_example}</code>
                  </pre>
                </div>
              </div>
            )}

            {finding.remediation.references.length > 0 && (
              <div className="references">
                <h3>References</h3>
                <ul className="reference-list">
                  {finding.remediation.references.map((ref, index) => (
                    <li key={index}>
                      <a href={ref} target="_blank" rel="noopener noreferrer">
                        {ref}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </section>
      )}

      {/* Timeline */}
      <section className="finding-section">
        <h2>Timeline</h2>
        <ul className="timeline">
          <li className="timeline-item">
            <span className="timeline-date">
              {new Date(finding.first_seen_at).toLocaleString()}
            </span>
            <span className="timeline-event">First detected</span>
          </li>
          {finding.fixed_at && (
            <li className="timeline-item">
              <span className="timeline-date">
                {new Date(finding.fixed_at).toLocaleString()}
              </span>
              <span className="timeline-event">Marked as fixed</span>
            </li>
          )}
          {finding.ignored_at && (
            <li className="timeline-item">
              <span className="timeline-date">
                {new Date(finding.ignored_at).toLocaleString()}
              </span>
              <span className="timeline-event">
                Ignored by {finding.ignored_by}
                {finding.ignore_reason && `: "${finding.ignore_reason}"`}
              </span>
            </li>
          )}
        </ul>
      </section>

      {/* Ignore Modal */}
      {showIgnoreModal && (
        <div className="modal-overlay" onClick={() => setShowIgnoreModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Ignore Finding</h3>
              <button className="close-btn" onClick={() => setShowIgnoreModal(false)}>×</button>
            </div>
            <div className="modal-body">
              <p>Please provide a reason for ignoring this finding:</p>
              <textarea
                className="form-textarea"
                value={ignoreReason}
                onChange={(e) => setIgnoreReason(e.target.value)}
                placeholder="e.g., Already mitigated at the infrastructure level..."
                rows={4}
              />
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowIgnoreModal(false)}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={() => updateStatusMutation.mutate({ status: 'ignored', reason: ignoreReason })}
                disabled={!ignoreReason.trim() || updateStatusMutation.isLoading}
              >
                Ignore Finding
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default FindingDetailPage
