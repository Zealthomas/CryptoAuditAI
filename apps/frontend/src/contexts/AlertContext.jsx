import React, { createContext, useContext, useState } from 'react'

const AlertContext = createContext()

export const useAlert = () => {
  const context = useContext(AlertContext)
  if (!context) {
    throw new Error('useAlert must be used within an AlertProvider')
  }
  return context
}

export const AlertProvider = ({ children }) => {
  const [alerts, setAlerts] = useState([])

  const showAlert = (message, type = 'error') => {
    const id = Math.random().toString(36).substring(2, 9)
    const newAlert = { id, message, type }
    
    setAlerts(prev => [...prev, newAlert])
    
    setTimeout(() => {
      removeAlert(id)
    }, 8000)
  }

  const removeAlert = (id) => {
    setAlerts(prev => prev.filter(alert => alert.id !== id))
  }

  return (
    <AlertContext.Provider value={{ alerts, showAlert, removeAlert }}>
      {children}
    </AlertContext.Provider>
  )
}