// src/App.jsx
import React, { useState, useEffect } from 'react'
import Header from './components/Header'
import AlertContainer from './components/AlertContainer'
import HeroSection from './components/HeroSection'
import AuditSection from './components/AuditSection'
import Dashboard from './components/Dashboard'
import ReportsSection from './components/ReportsSection'
import AlertsSection from './components/AlertsSection'
import Footer from './components/Footer'
import ChatPanel from './components/ChatPanel'

import { AlertProvider } from './contexts/AlertContext'
import { ApiProvider } from './contexts/ApiContext'
import { AuditProvider, useAudit } from './contexts/AuditContext'

import './App.css'

// ----------------------------
// AppLayout — separated for clean context hierarchy
// ----------------------------
const AppLayout = () => {
  const [activeSection, setActiveSection] = useState('dashboard')
  const { auditRefreshCount, lastAuditTime } = useAudit()

  // Optional: respond to audit refresh triggers globally
  useEffect(() => {
    if (auditRefreshCount > 0) {
      console.log(`🔄 Audit dashboard refreshed at ${lastAuditTime}`)
    }
  }, [auditRefreshCount, lastAuditTime])

  return (
    <div className="app">
      <Header activeSection={activeSection} setActiveSection={setActiveSection} />
      <AlertContainer />

      <main className="main">
        <div className="container">
          <HeroSection />
          <AuditSection />

          {activeSection === 'dashboard' && <Dashboard key={auditRefreshCount} />}
          {activeSection === 'reports' && <ReportsSection />}
          {activeSection === 'alerts' && <AlertsSection />}
        </div>
      </main>

      <Footer />
      <ChatPanel />
    </div>
  )
}

// ----------------------------
// Root App Wrapper
// ----------------------------
const App = () => {
  return (
    <AlertProvider>
      <ApiProvider>
        <AuditProvider>
          <AppLayout />
        </AuditProvider>
      </ApiProvider>
    </AlertProvider>
  )
}

export default App