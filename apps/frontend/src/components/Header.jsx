import React from 'react'
import { useApi } from '../contexts/ApiContext'

const Header = ({ activeSection, setActiveSection }) => {
  const { isBackendConnected } = useApi()
  
  const getConnectionStatus = () => {
    if (isBackendConnected) {
      return { class: 'status-connected', text: 'Connected' }
    }
    return { class: 'status-disconnected', text: 'Disconnected' }
  }
  
  const status = getConnectionStatus()

  return (
    <header className="header">
      <div className="container">
        <nav className="nav">
          <a href="#" className="logo">CryptoAuditAI</a>
          <ul className="nav-links">
            <li>
              <a 
                href="#" 
                className={activeSection === 'dashboard' ? 'active' : ''}
                onClick={(e) => { e.preventDefault(); setActiveSection('dashboard'); }}
              >
                Dashboard
              </a>
            </li>
            <li>
              <a 
                href="#" 
                className={activeSection === 'reports' ? 'active' : ''}
                onClick={(e) => { e.preventDefault(); setActiveSection('reports'); }}
              >
                Reports
              </a>
            </li>
            <li>
              <a 
                href="#" 
                className={activeSection === 'alerts' ? 'active' : ''}
                onClick={(e) => { e.preventDefault(); setActiveSection('alerts'); }}
              >
                Alerts
              </a>
            </li>
          </ul>
          <div className="user-menu">
            <div id="connectionStatus">
              <span className={`status-indicator ${status.class}`}></span>
              <span>{status.text}</span>
            </div>
          </div>
        </nav>
      </div>
    </header>
  )
}

export default Header