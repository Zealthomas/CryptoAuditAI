// src/components/AuditSection.jsx
import React, { useState } from 'react'
import { useApi } from '../contexts/ApiContext'
import { useAlert } from '../contexts/AlertContext'
import { useAudit } from '../contexts/AuditContext'
import { Zap, Shield, Download } from 'lucide-react'

const AuditSection = () => {
  const [address, setAddress] = useState('')
  const [exchange, setExchange] = useState('')
  const [limit, setLimit] = useState(100)
  const [loading, setLoading] = useState(false)
  const [analysisStatus, setAnalysisStatus] = useState('hidden')
  const [statusText, setStatusText] = useState('')
  const [analysisResults, setAnalysisResults] = useState(null)
  const [riskScore, setRiskScore] = useState(0)

  const { apiCall, currentUserId } = useApi()
  const { showAlert } = useAlert()
  const { triggerDashboardRefresh } = useAudit()

  const validateEthereumAddress = (addr) => {
    return /^0x[a-fA-F0-9]{40}$/.test(addr)
  }

  const handleIngest = async (e) => {
    e.preventDefault()

    if (!address && !exchange) {
      showAlert('Please enter a wallet address or select an exchange', 'error')
      return
    }

    if (address && !validateEthereumAddress(address)) {
      showAlert('Please enter a valid Ethereum address (0x followed by 40 hex characters)', 'error')
      return
    }

    setLoading(true)
    setAnalysisStatus('visible')
    setStatusText('Ingesting wallet transactions...')

    try {
      const payload = {
        user_id: currentUserId,
        limit: limit
      }

      if (address) payload.wallet_address = address
      if (exchange) payload.exchange_name = exchange

      const ingestResponse = await apiCall('/ingest', {
        method: 'POST',
        body: JSON.stringify(payload)
      })

      console.log('Ingestion started:', ingestResponse)
      showAlert('Transaction ingestion started successfully', 'success')

      // Step 2: Run audit
      setStatusText('Running AI-powered risk analysis...')

      const auditResponse = await apiCall('/audit', {
        method: 'POST',
        body: JSON.stringify({
          user_id: currentUserId,
          report_type: 'comprehensive'
        })
      })

      console.log('Audit started:', auditResponse)
      showAlert('AI audit analysis started successfully', 'success')

      // Step 3: Wait and fetch results
      setStatusText('Processing results...')
      await new Promise(resolve => setTimeout(resolve, 3000))

      // Get updated stats
      const stats = await apiCall(`/users/${currentUserId}/stats`)
      
      const riskPercentage = stats.total_transactions > 0 
        ? Math.min(100, Math.round((stats.high_risk_transactions / stats.total_transactions) * 100))
        : 0

      setRiskScore(riskPercentage)

      const riskLevel = riskPercentage < 30 ? 'low' : riskPercentage < 70 ? 'medium' : 'high'

      setAnalysisResults({
        riskScore: riskPercentage,
        riskLevel: riskLevel,
        totalTransactions: stats.total_transactions,
        highRiskTransactions: stats.high_risk_transactions
      })

      setAnalysisStatus('hidden')
      
      // Trigger dashboard refresh
      triggerDashboardRefresh()

      showAlert('Audit completed successfully!', 'success')

      // Reset form
      setAddress('')
      setExchange('')
      setLimit(100)

    } catch (error) {
      console.error('Audit failed:', error)
      showAlert(`Audit failed: ${error.message}`, 'error')
      setAnalysisStatus('hidden')
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="audit-section">
      <h2 style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
        <Zap size={24} />
        Quick Audit
      </h2>

      <form className="audit-form" onSubmit={handleIngest}>
        <div className="input-group">
          <label htmlFor="address">Wallet Address</label>
          <input
            type="text"
            id="address"
            placeholder="0x742d35Cc6634C0532925a3b8D03a68cf9b07c93c"
            value={address}
            onChange={(e) => setAddress(e.target.value.trim())}
          />
        </div>

        <div className="input-group">
          <label htmlFor="exchange">Exchange (Optional)</label>
          <select
            id="exchange"
            value={exchange}
            onChange={(e) => setExchange(e.target.value)}
          >
            <option value="">Select Exchange</option>
            <option value="binance">Binance</option>
            <option value="coinbase">Coinbase</option>
            <option value="kraken">Kraken</option>
            <option value="bybit">Bybit</option>
          </select>
        </div>

        <div className="input-group">
          <label htmlFor="limit">Transaction Limit</label>
          <input
            type="number"
            id="limit"
            min="10"
            max="1000"
            value={limit}
            onChange={(e) => setLimit(parseInt(e.target.value))}
          />
        </div>

        <button type="submit" className="btn" disabled={loading}>
          {loading ? (
            <>
              <div className="spinner" style={{ width: '16px', height: '16px', marginRight: '8px' }} />
              Analyzing...
            </>
          ) : (
            <>
              <Download size={16} style={{ marginRight: '4px' }} />
              Start Audit
            </>
          )}
        </button>
      </form>

      {/* Analysis Status */}
      {analysisStatus === 'visible' && (
        <div style={{ marginTop: '30px' }}>
          <div className="loading">
            <div className="spinner"></div>
            <span>{statusText}</span>
          </div>
        </div>
      )}

      {/* Analysis Results */}
      {analysisResults && (
        <div style={{ marginTop: '40px', padding: '30px', background: 'var(--bg-secondary)', borderRadius: '12px', border: '1px solid var(--border)' }}>
          <h3 style={{ marginBottom: '24px', color: 'var(--accent-blue)', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Shield size={20} />
            Risk Assessment
          </h3>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '30px' }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ 
                fontSize: '3rem', 
                fontWeight: '800', 
                marginBottom: '10px',
                color: analysisResults.riskLevel === 'low' ? 'var(--accent-green)' : 
                       analysisResults.riskLevel === 'medium' ? 'var(--accent-orange)' : 
                       'var(--accent-red)'
              }}>
                {riskScore}
              </div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '20px' }}>
                Overall Risk Score
              </div>

              <div className={`risk-badge ${analysisResults.riskLevel}`}>
                {analysisResults.riskLevel}
              </div>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <div style={{ marginBottom: '20px' }}>
                <div style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '8px' }}>
                  Total Transactions Analyzed
                </div>
                <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--accent-blue)' }}>
                  {analysisResults.totalTransactions}
                </div>
              </div>

              <div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '14px', marginBottom: '8px' }}>
                  High Risk Transactions
                </div>
                <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--accent-red)' }}>
                  {analysisResults.highRiskTransactions}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}

export default AuditSection