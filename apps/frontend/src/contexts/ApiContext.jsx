import React, { createContext, useContext, useState, useEffect } from 'react'
import { useAlert } from './AlertContext'

const ApiContext = createContext()

export const useApi = () => {
  const context = useContext(ApiContext)
  if (!context) {
    throw new Error('useApi must be used within an ApiProvider')
  }
  return context
}

export const ApiProvider = ({ children }) => {
  const API_BASE_URL = 'http://localhost:8000'
  const CURRENT_USER_ID = 1
  const [isBackendConnected, setIsBackendConnected] = useState(false)
  const [aiChatEnabled, setAiChatEnabled] = useState(false)
  const { showAlert } = useAlert()

  const apiCall = async (endpoint, options = {}) => {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          ...options.headers,
        },
        ...options,
      })

      let data
      const contentType = response.headers.get('content-type')
      
      if (contentType && contentType.includes('application/json')) {
        data = await response.json()
      } else {
        const text = await response.text()
        data = { message: text }
      }

      if (!response.ok) {
        throw new Error(data.detail || data.message || `HTTP ${response.status}: ${response.statusText}`)
      }

      return data
    } catch (error) {
      console.error(`API call failed for ${endpoint}:`, error)
      throw error
    }
  }

  const testBackendConnection = async () => {
    try {
      const health = await apiCall('/health')
      setIsBackendConnected(true)
      setAiChatEnabled(health.ollama_configured)
    } catch (error) {
      console.error('Backend connection failed:', error)
      setIsBackendConnected(false)
      setAiChatEnabled(false)
    }
  }

  useEffect(() => {
    testBackendConnection()
  }, [])

  return (
    <ApiContext.Provider value={{
      isBackendConnected,
      aiChatEnabled,
      testBackendConnection,
      apiCall,
      currentUserId: CURRENT_USER_ID,
    }}>
      {children}
    </ApiContext.Provider>
  )
}