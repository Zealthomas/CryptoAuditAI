import React, { useState, useEffect } from 'react'
import { useApi } from '../contexts/ApiContext'
import { useAlert } from '../contexts/AlertContext'
import { AlertCircle, CheckCircle, RefreshCw, Clock } from 'lucide-react'

const AlertsSection = () => {
  const [alerts, setAlerts] = useState([])
  const { apiCall, currentUserId } = useApi()
  const { showAlert } = useAlert()

  useEffect(() => {
    loadUserAlerts()
  }, [])

  const loadUserAlerts = async () => {
    try {
      const alertsData = await apiCall(`/users/${currentUserId}/alerts?limit=50`)
      setAlerts(alertsData || [])
    } catch (error) {
      console.error('Failed to load alerts:', error)
    }
  }

  const resolveAlert = async (alertId) => {
    try {
      await apiCall(`/users/${currentUserId}/alerts/${alertId}/resolve`, {
        method: 'POST'
      })
      showAlert('Alert resolved successfully', 'success')
      await loadUserAlerts()
    } catch (error) {
      console.error('Failed to resolve alert:', error)
      showAlert(`Failed to resolve alert: ${error.message}`, 'error')
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
        <h3><AlertCircle size={20} style={{ marginRight: '8px', display: 'inline' }} /> Security Alerts</h3>
        <button className="btn btn-secondary" onClick={loadUserAlerts}>
          <RefreshCw size={16} style={{ marginRight: '4px', display: 'inline' }} /> Refresh
        </button>
      </div>

      <div>
        {alerts.length === 0 ? (
          <p className="empty-state-message">
            No security alerts at this time. Your crypto assets appear secure.
          </p>
        ) : (
          alerts.map(alert => (
            <div 
              key={alert.id} 
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
                padding: '20px',
                marginBottom: '16px',
                background: alert.severity === 'critical' ? 'rgba(255, 71, 87, 0.1)' :
                           alert.severity === 'high' ? 'rgba(255, 149, 0, 0.1)' :
                           alert.severity === 'warning' ? 'rgba(255, 193, 7, 0.1)' : 
                           'rgba(0, 212, 255, 0.1)',
                border: alert.severity === 'critical' ? '1px solid rgba(255, 71, 87, 0.2)' :
                        alert.severity === 'high' ? '1px solid rgba(255, 149, 0, 0.2)' :
                        alert.severity === 'warning' ? '1px solid rgba(255, 193, 7, 0.2)' : 
                        '1px solid rgba(0, 212, 255, 0.2)',
                borderRadius: '12px',
                borderLeft: alert.severity === 'critical' ? '4px solid var(--accent-red)' :
                           alert.severity === 'high' ? '4px solid var(--accent-orange)' :
                           alert.severity === 'warning' ? '4px solid var(--accent-orange)' : 
                           '4px solid var(--accent-blue)'
              }}
            >
              <div style={{ flex: 1 }}>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '12px', 
                  marginBottom: '8px' 
                }}>
                  <div style={{ fontWeight: 600, fontSize: '16px', color: 'var(--text-primary)' }}>
                    {alert.title}
                  </div>
                  <div className={`risk-badge ${alert.severity}`}>
                    {alert.severity}
                  </div>
                </div>
                
                <div style={{ 
                  color: 'var(--text-secondary)', 
                  marginBottom: '12px',
                  lineHeight: 1.5
                }}>
                  {alert.message}
                </div>
                
                <div style={{ 
                  display: 'flex', 
                  gap: '16px', 
                  fontSize: '12px', 
                  color: 'var(--text-muted)' 
                }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                    <Clock size={12} />
                    {new Date(alert.created_at).toLocaleString()}
                  </span>
                  <span>Type: {alert.alert_type}</span>
                  {alert.is_resolved && (
                    <span style={{ 
                      color: 'var(--accent-green)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}>
                      <CheckCircle size={12} /> Resolved
                    </span>
                  )}
                </div>
              </div>

              <div style={{ marginLeft: '16px' }}>
                {!alert.is_resolved ? (
                  <button
                    className="btn btn-secondary"
                    style={{ padding: '8px 16px', fontSize: '12px' }}
                    onClick={() => resolveAlert(alert.id)}
                  >
                    <CheckCircle size={14} style={{ marginRight: '4px', display: 'inline' }} /> Resolve
                  </button>
                ) : (
                  <div style={{ 
                    color: 'var(--accent-green)', 
                    fontSize: '14px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px'
                  }}>
                    <CheckCircle size={14} />
                    Resolved
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  )
}

export default AlertsSection