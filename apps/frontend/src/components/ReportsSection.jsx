import React, { useState, useEffect } from 'react'
import { useApi } from '../contexts/ApiContext'
import { useAlert } from '../contexts/AlertContext'
import { FileText, Download, Calendar, RefreshCw, BarChart3 } from 'lucide-react'

const ReportsSection = () => {
  const [reports, setReports] = useState([])
  const [loading, setLoading] = useState(false)
  const { apiCall, currentUserId } = useApi()
  const { showAlert } = useAlert()

  useEffect(() => {
    loadUserReports()
  }, [])

  const loadUserReports = async () => {
    try {
      const reportsData = await apiCall(`/users/${currentUserId}/reports`)
      setReports(reportsData || [])
    } catch (error) {
      console.error('Failed to load reports:', error)
    }
  }

  const generateReport = async () => {
    setLoading(true)
    try {
      showAlert('Generating comprehensive audit report...', 'warning')
     
      await apiCall('/report', {
        method: 'POST',
        body: JSON.stringify({
          user_id: currentUserId,
          report_type: 'comprehensive'
        })
      })
      
      showAlert('Audit report generated successfully!', 'success')
      await loadUserReports()
    } catch (error) {
      console.error('Failed to generate report:', error)
      showAlert(`Failed to generate report: ${error.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const downloadReport = async (reportId, format) => {
    try {
  const downloadUrl = `/api/reports/${reportId}/download?format=${format}`
  const response = await fetch(downloadUrl)
     
      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`)
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `audit_report_${reportId}.${format}`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      window.URL.revokeObjectURL(url)

      showAlert('Report downloaded successfully!', 'success')
    } catch (error) {
      console.error('Download failed:', error)
      showAlert(`Download failed: ${error.message}`, 'error')
    }
  }

  return (
    <section className="card">
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '24px' 
      }}>
        <h3><FileText size={20} /> Audit Reports</h3>
        <div style={{ display: 'flex', gap: '12px' }}>
          <button className="btn btn-secondary" onClick={loadUserReports}>
            <RefreshCw size={16} /> Refresh
          </button>
          <button className="btn" onClick={generateReport} disabled={loading}>
            {loading ? (
              <>
                <div className="spinner" style={{ width: '16px', height: '16px', marginRight: '8px' }} />
                Generating...
              </>
            ) : (
              <>
                <FileText size={16} /> Generate New Report
              </>
            )}
          </button>
        </div>
      </div>

      <div>
        {reports.length === 0 ? (
          <p className="empty-state-message">
            No reports generated yet. Run an audit to create your first report.
          </p>
        ) : (
          reports.map(report => (
            <div key={report.id} className="transaction-item">
              <div className="transaction-info">
                <div style={{ fontWeight: 600, fontSize: '16px', marginBottom: '8px' }}>
                  {report.title || 'Audit Report'}
                </div>
                <div style={{ 
                  display: 'flex', 
                  gap: '16px', 
                  fontSize: '14px', 
                  color: 'var(--text-secondary)',
                  marginBottom: '8px' 
                }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Calendar size={14} /> 
                    {new Date(report.created_at).toLocaleDateString()}
                  </span>
                  <span>Type: {report.report_type || 'comprehensive'}</span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <BarChart3 size={14} />
                    Transactions: {report.total_transactions || 0}
                  </span>
                </div>
                <div style={{ display: 'flex', gap: '16px', fontSize: '12px' }}>
                  <span style={{ 
                    color: report.overall_risk_score > 70 ? 'var(--accent-red)' : 
                           report.overall_risk_score > 40 ? 'var(--accent-orange)' : 
                           'var(--accent-green)'
                  }}>
                    Risk Score: {report.overall_risk_score || 0}/100
                  </span>
                  <span style={{ color: 'var(--accent-green)' }}>
                    Compliance: {report.compliance_score || 0}/100
                  </span>
                  {report.flagged_transactions > 0 && (
                    <span style={{ color: 'var(--accent-red)' }}>
                      {report.flagged_transactions} flagged
                    </span>
                  )}
                </div>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => downloadReport(report.id, 'pdf')}
                  title="Download PDF"
                >
                  <Download size={16} /> PDF
                </button>
                <button
                  className="btn btn-secondary"
                  onClick={() => downloadReport(report.id, 'json')}
                  title="Download JSON"
                >
                  <Download size={16} /> JSON
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  )
}

export default ReportsSection