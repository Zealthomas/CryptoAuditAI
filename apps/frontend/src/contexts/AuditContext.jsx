// src/contexts/AuditContext.jsx
import React, { createContext, useContext, useState } from 'react'

const AuditContext = createContext()

export const useAudit = () => {
  const context = useContext(AuditContext)
  if (!context) {
    throw new Error('useAudit must be used within an AuditProvider')
  }
  return context
}

export const AuditProvider = ({ children }) => {
  const [auditRefreshCount, setAuditRefreshCount] = useState(0)
  const [lastAuditTime, setLastAuditTime] = useState(null)

  const triggerDashboardRefresh = () => {
    setAuditRefreshCount(prev => prev + 1)
    setLastAuditTime(new Date())
    console.log('Dashboard refresh triggered')
  }

  return (
    <AuditContext.Provider value={{ auditRefreshCount, triggerDashboardRefresh, lastAuditTime }}>
      {children}
    </AuditContext.Provider>
  )
}