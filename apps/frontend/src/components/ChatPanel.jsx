import React, { useState, useEffect, useRef } from 'react'
import { useApi } from '../contexts/ApiContext'

const ChatPanel = () => {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState([
    { sender: 'ai', text: "Hello! I'm your crypto audit AI assistant. Ask me about your transactions, risk assessments, or compliance status." }
  ])
  const [inputText, setInputText] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef(null)
  const { apiCall, aiChatEnabled, isBackendConnected } = useApi()

  const toggleChat = () => {
    setIsOpen(!isOpen)
    if (!isOpen) {
      const toggleBtn = document.querySelector('.chat-toggle')
      if (toggleBtn) toggleBtn.classList.remove('has-message')
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isTyping])

  const sendMessage = async (e) => {
    e.preventDefault()
    const msg = inputText.trim()
    if (!msg) return

    if (!isBackendConnected) {
      addMessage('ai', 'Backend is not connected. Please wait for the connection to be established.')
      return
    }

    if (!aiChatEnabled) {
      addMessage('ai', 'AI chat is not available. The backend needs to be configured with Ollama/Mistral.')
      return
    }

    setInputText('')
    addMessage('user', msg)
    setIsTyping(true)

    try {
      const response = await apiCall('/chat', {
        method: 'POST',
        body: JSON.stringify({ 
          query: msg,
          user_id: 1,
          context: {
            timestamp: new Date().toISOString(),
          }
        })
      })

      setIsTyping(false)

      if (response.response && response.response.trim()) {
        addMessage('ai', response.response)
        if (!isOpen) {
          const toggleBtn = document.querySelector('.chat-toggle')
          if (toggleBtn) toggleBtn.classList.add('has-message')
        }
      } else if (response.error) {
        addMessage('ai', `I encountered an error: ${response.error}`)
      } else {
        addMessage('ai', "I received your message but couldn't generate a response. Please try again.")
      }
    } catch (error) {
      setIsTyping(false)
      console.error('Chat error:', error)
      
      let errorMsg = "I'm having trouble responding right now. "
      
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        errorMsg += 'Please check your internet connection and try again.'
      } else if (error.message.includes('404')) {
        errorMsg += 'The chat service endpoint is not available.'
      } else if (error.message.includes('500')) {
        errorMsg += "There's a server error. The AI model might be unavailable."
      } else if (error.message.includes('timeout')) {
        errorMsg += 'The request timed out. The AI model might be processing other requests.'
      } else {
        errorMsg += `Error: ${error.message}`
      }
      
      addMessage('ai', errorMsg)
    }
  }

  const addMessage = (sender, text) => {
    setMessages(prev => [...prev, { sender, text }])
  }

  return (
    <>
      <button className="chat-toggle" onClick={toggleChat} title="Chat with AI Assistant">💬</button>

      <div className="chat-panel" style={{display: isOpen ? 'flex' : 'none'}}>
        <div className="chat-header">
          <h3>
            <span>AI Assistant</span>
            <div className={`ai-status ${aiChatEnabled ? 'online' : 'offline'}`}>
              {aiChatEnabled ? 'AI Ready' : 'AI Offline'}
            </div>
          </h3>
          <button onClick={toggleChat} title="Close chat">&times;</button>
        </div>
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.sender}`}>
              {msg.text}
              {msg.sender === 'ai' && (
                <div style={{fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px', textAlign: 'right'}}>
                  {new Date().toLocaleTimeString()}
                </div>
              )}
            </div>
          ))}
          {isTyping && (
            <div className="typing">
              <span>AI is thinking</span>
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <form className="chat-input" onSubmit={sendMessage}>
          <input 
            type="text" 
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Ask about your audit results..." 
            required 
            maxLength={500} 
          />
          <button type="submit">Send</button>
        </form>
      </div>
    </>
  )
}

export default ChatPanel