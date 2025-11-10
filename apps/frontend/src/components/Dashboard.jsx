// src/components/Dashboard.jsx
import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { useApi } from '../contexts/ApiContext'
import { useAlert } from '../contexts/AlertContext'
import { useAudit } from '../contexts/AuditContext'
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar, Area, AreaChart, Legend
} from 'recharts'
import { 
  Activity, AlertTriangle, Shield, TrendingUp, Users, 
  RefreshCw, Copy, BarChart3, Clock, Wallet
} from 'lucide-react'

const RISK_COLORS = {
  low: '#00ff88',
  medium: '#ff9500',
  high: '#ff4757',
  critical: '#ff4757'
}

const Dashboard = () => {
  const { apiCall, currentUserId } = useApi()
  const { showAlert } = useAlert()
  const { auditRefreshCount, lastAuditTime } = useAudit()
  
  const [allTransactions, setAllTransactions] = useState([])
  const [wallets, setWallets] = useState([])
  const [selectedWallet, setSelectedWallet] = useState('all')
  const [riskFilter, setRiskFilter] = useState('all')
  const [loading, setLoading] = useState(false)
  const [statsLoading, setStatsLoading] = useState(false)

  // Load all transactions
  const loadAllTransactions = useCallback(async () => {
    try {
      setLoading(true)
      const data = await apiCall(`/users/${currentUserId}/transactions?limit=1000`)
      setAllTransactions(data || [])
      
      // Extract unique wallet addresses from BOTH from_address and to_address
      const allAddresses = new Set()
      data.forEach(tx => {
        if (tx.from_address && tx.from_address !== null) {
          allAddresses.add(tx.from_address)
        }
        if (tx.to_address && tx.to_address !== null) {
          allAddresses.add(tx.to_address)
        }
      })
      
      const uniqueWallets = Array.from(allAddresses)
      setWallets(uniqueWallets)
      
      console.log('📋 Loaded', data?.length || 0, 'transactions,', uniqueWallets.length, 'unique addresses')
    } catch (error) {
      console.error('Failed to load transactions:', error)
      setAllTransactions([])
      showAlert('Failed to load transactions', 'error')
    } finally {
      setLoading(false)
    }
  }, [apiCall, currentUserId, showAlert])

  // Filter transactions by selected wallet and risk level
  const filteredTransactions = useMemo(() => {
    let filtered = allTransactions

    // Filter by wallet
    if (selectedWallet !== 'all') {
      filtered = filtered.filter(tx => 
        tx.from_address === selectedWallet || tx.to_address === selectedWallet
      )
    }

    // Filter by risk level
    if (riskFilter !== 'all') {
      filtered = filtered.filter(tx => tx.risk_level === riskFilter)
    }

    console.log('🔍 Filtered:', filtered.length, 'transactions for wallet:', selectedWallet)
    return filtered
  }, [allTransactions, selectedWallet, riskFilter])

  // Calculate stats based on filtered transactions
  const stats = useMemo(() => {
    const total = filteredTransactions.length
    const highRisk = filteredTransactions.filter(tx => 
      tx.risk_level === 'high' || tx.risk_level === 'critical'
    ).length
    const anomalies = filteredTransactions.filter(tx => tx.anomaly_detected).length
    
    // Count unique wallets in FILTERED transactions only
    const filteredWallets = new Set()
    filteredTransactions.forEach(tx => {
      if (tx.from_address) filteredWallets.add(tx.from_address)
      if (tx.to_address) filteredWallets.add(tx.to_address)
    })

    return {
      total_transactions: total,
      high_risk_transactions: highRisk,
      unresolved_alerts: anomalies,
      total_wallets: selectedWallet === 'all' ? wallets.length : filteredWallets.size
    }
  }, [filteredTransactions, wallets.length, selectedWallet])

  // Load data on mount and when audit completes
  useEffect(() => {
    console.log('🔄 Loading dashboard data... (Refresh count:', auditRefreshCount, ')')
    loadAllTransactions()
  }, [loadAllTransactions, auditRefreshCount])

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      console.log('⏰ Auto-refreshing dashboard...')
      loadAllTransactions()
    }, 30000)

    return () => clearInterval(interval)
  }, [loadAllTransactions])

  // Process chart data from filtered transactions
  const chartData = useMemo(() => {
    const timeSeriesMap = {}
    filteredTransactions.forEach(tx => {
      const date = new Date(tx.timestamp).toLocaleDateString()
      if (!timeSeriesMap[date]) {
        timeSeriesMap[date] = { date, transactions: 0, volume: 0, highRisk: 0 }
      }
      timeSeriesMap[date].transactions += 1
      timeSeriesMap[date].volume += tx.amount || 0
      if (tx.risk_level === 'high' || tx.risk_level === 'critical') {
        timeSeriesMap[date].highRisk += 1
      }
    })
    
    const timeSeries = Object.values(timeSeriesMap).sort(
      (a, b) => new Date(a.date) - new Date(b.date)
    )

    const riskDist = { low: 0, medium: 0, high: 0, critical: 0 }
    filteredTransactions.forEach(tx => {
      const risk = tx.risk_level || 'low'
      riskDist[risk] = (riskDist[risk] || 0) + 1
    })
    
    const riskDistribution = Object.entries(riskDist)
      .filter(([_, count]) => count > 0)
      .map(([risk, count]) => ({
        name: risk.charAt(0).toUpperCase() + risk.slice(1),
        value: count,
        color: RISK_COLORS[risk]
      }))

    const typeDist = {}
    filteredTransactions.forEach(tx => {
      const type = tx.transaction_type || 'unknown'
      typeDist[type] = (typeDist[type] || 0) + 1
    })
    
    const typeDistribution = Object.entries(typeDist).map(([type, count]) => ({
      name: type.charAt(0).toUpperCase() + type.slice(1),
      value: count
    }))

    const counterpartyMap = {}
    filteredTransactions.forEach(tx => {
      // Consider both from and to addresses as counterparties
      const addresses = []
      if (tx.from_address && tx.from_address !== selectedWallet) {
        addresses.push(tx.from_address)
      }
      if (tx.to_address && tx.to_address !== selectedWallet) {
        addresses.push(tx.to_address)
      }

      addresses.forEach(address => {
        if (!counterpartyMap[address]) {
          counterpartyMap[address] = {
            address,
            transactions: 0,
            totalRisk: 0,
            avgRisk: 0
          }
        }
        counterpartyMap[address].transactions += 1
        counterpartyMap[address].totalRisk += tx.risk_score || 0
        counterpartyMap[address].avgRisk = 
          counterpartyMap[address].totalRisk / counterpartyMap[address].transactions
      })
    })
    
    const counterpartyRisk = Object.values(counterpartyMap)
      .sort((a, b) => b.avgRisk - a.avgRisk)
      .slice(0, 10)

    return { timeSeries, riskDistribution, typeDistribution, counterpartyRisk }
  }, [filteredTransactions, selectedWallet])

  const complianceScore = useMemo(() => {
    if (stats.total_transactions === 0) return 0
    return Math.round(
      ((stats.total_transactions - stats.high_risk_transactions) / 
       stats.total_transactions) * 100
    )
  }, [stats])

  const copyToClipboard = (text) => {
    navigator.clipboard?.writeText(text)
    showAlert('Copied to clipboard', 'success')
  }

  const handleManualRefresh = () => {
    showAlert('Refreshing dashboard...', 'warning')
    loadAllTransactions()
  }

  return (
    <>
      {/* Header with filters and refresh controls */}
      <div style={{ 
        marginBottom: '20px', 
        display: 'flex', 
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          {/* Wallet Filter */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Wallet size={16} style={{ color: 'var(--accent-blue)' }} />
            <select 
              className="input-group" 
              style={{ padding: '8px 12px', width: 'auto', minWidth: '200px' }} 
              value={selectedWallet} 
              onChange={(e) => setSelectedWallet(e.target.value)}
            >
              <option value="all">All Wallets ({wallets.length})</option>
              {wallets.map(wallet => (
                <option key={wallet} value={wallet}>
                  {`${wallet.slice(0, 8)}...${wallet.slice(-6)}`}
                </option>
              ))}
            </select>
          </div>

          <button 
            className="btn btn-secondary" 
            onClick={handleManualRefresh}
            disabled={statsLoading || loading}
          >
            <RefreshCw size={16} style={{ marginRight: '4px' }} /> 
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
          
        {lastAuditTime && (
          <div style={{ 
            fontSize: '12px', 
            color: 'var(--text-muted)',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}>
            <Clock size={14} />
            Last audit: {lastAuditTime.toLocaleTimeString()}
          </div>
        )}
      </div>

      {/* Wallet Filter Info Banner */}
      {selectedWallet !== 'all' && (
        <div style={{
          background: 'rgba(0, 212, 255, 0.1)',
          border: '1px solid rgba(0, 212, 255, 0.3)',
          borderRadius: '8px',
          padding: '12px 16px',
          marginBottom: '20px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Shield size={16} style={{ color: 'var(--accent-blue)' }} />
            <span style={{ fontSize: '14px' }}>
              Showing data for wallet: <code style={{ 
                background: 'rgba(0, 212, 255, 0.2)', 
                padding: '2px 8px', 
                borderRadius: '4px',
                fontFamily: 'monospace'
              }}>
                {`${selectedWallet.slice(0, 12)}...${selectedWallet.slice(-8)}`}
              </code>
            </span>
          </div>
          <button 
            className="btn btn-secondary" 
            style={{ padding: '6px 12px', fontSize: '12px' }}
            onClick={() => setSelectedWallet('all')}
          >
            Clear Filter
          </button>
        </div>
      )}

      {/* Stats Cards */}
      <section className="dashboard-grid">
        <div className="card stat-card">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity size={20} /> Total Transactions
          </h3>
          <div className="stat-number">
            {loading ? (
              <div className="spinner" style={{ width: '40px', height: '40px' }} />
            ) : (
              stats.total_transactions
            )}
          </div>
          <div className="stat-label">
            {selectedWallet === 'all' ? 'All wallets' : 'This wallet'}
          </div>
        </div>

        <div className="card stat-card">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertTriangle size={20} /> High Risk
          </h3>
          <div className="stat-number risk-high">
            {loading ? (
              <div className="spinner" style={{ width: '40px', height: '40px' }} />
            ) : (
              <>
                {stats.high_risk_transactions} ({stats.total_transactions > 0 
                  ? Math.round((stats.high_risk_transactions / stats.total_transactions) * 100) 
                  : 0}%)
              </>
            )}
          </div>
          <div className="stat-label">Flagged transactions</div>
        </div>

        <div className="card stat-card">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Shield size={20} /> Compliance Score
          </h3>
          <div className="stat-number risk-low">
            {loading ? (
              <div className="spinner" style={{ width: '40px', height: '40px' }} />
            ) : (
              `${complianceScore}%`
            )}
          </div>
          <div className="stat-label">Regulatory compliance</div>
        </div>

        <div className="card stat-card">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Wallet size={20} /> {selectedWallet === 'all' ? 'Total Wallets' : 'Counterparties'}
          </h3>
          <div className="stat-number" style={{ color: 'var(--accent-blue)' }}>
            {loading ? (
              <div className="spinner" style={{ width: '40px', height: '40px' }} />
            ) : (
              stats.total_wallets
            )}
          </div>
          <div className="stat-label">
            {selectedWallet === 'all' ? 'Unique addresses' : 'Connected addresses'}
          </div>
        </div>
      </section>

      {/* Charts Grid */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', 
        gap: '30px', 
        marginBottom: '40px' 
      }}>
        {/* Time Series Chart */}
        <section className="card">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingUp size={20} /> Transaction Activity
          </h3>
          {chartData.timeSeries.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData.timeSeries}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="date" stroke="var(--text-secondary)" fontSize={12} />
                <YAxis stroke="var(--text-secondary)" fontSize={12} />
                <Tooltip 
                  contentStyle={{ 
                    background: 'var(--bg-card)', 
                    border: '1px solid var(--border)', 
                    borderRadius: '8px' 
                  }} 
                />
                <Area 
                  type="monotone" 
                  dataKey="transactions" 
                  stroke="var(--accent-blue)" 
                  fill="rgba(0, 212, 255, 0.2)" 
                  strokeWidth={2} 
                  name="Transactions" 
                />
                <Area 
                  type="monotone" 
                  dataKey="highRisk" 
                  stroke="var(--accent-red)" 
                  fill="rgba(255, 71, 87, 0.2)" 
                  strokeWidth={2} 
                  name="High Risk" 
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <p className="empty-state-message">No transaction data available</p>
            </div>
          )}
        </section>

        {/* Risk Distribution */}
        <section className="card">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Shield size={20} /> Risk Distribution
          </h3>
          {chartData.riskDistribution.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie 
                  data={chartData.riskDistribution} 
                  cx="50%" 
                  cy="50%" 
                  innerRadius={60} 
                  outerRadius={120} 
                  paddingAngle={5} 
                  dataKey="value"
                >
                  {chartData.riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    background: 'var(--bg-card)', 
                    border: '1px solid var(--border)', 
                    borderRadius: '8px' 
                  }} 
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <p className="empty-state-message">No risk data available</p>
            </div>
          )}
        </section>

        {/* Transaction Types */}
        <section className="card">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart3 size={20} /> Transaction Types
          </h3>
          {chartData.typeDistribution.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData.typeDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis dataKey="name" stroke="var(--text-secondary)" fontSize={12} />
                <YAxis stroke="var(--text-secondary)" fontSize={12} />
                <Tooltip 
                  contentStyle={{ 
                    background: 'var(--bg-card)', 
                    border: '1px solid var(--border)', 
                    borderRadius: '8px' 
                  }} 
                />
                <Bar dataKey="value" fill="var(--accent-blue)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <p className="empty-state-message">No transaction type data available</p>
            </div>
          )}
        </section>

        {/* Counterparty Risk */}
        <section className="card">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Users size={20} /> Counterparty Risk
          </h3>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {chartData.counterpartyRisk.length > 0 ? (
              chartData.counterpartyRisk.map((party, index) => (
                <div key={index} className="transaction-item">
                  <div className="transaction-info">
                    <div className="transaction-hash">
                      {`${party.address.slice(0, 8)}...${party.address.slice(-6)}`}
                      <button 
                        style={{ 
                          background: 'none', 
                          border: 'none', 
                          color: 'var(--text-muted)', 
                          cursor: 'pointer', 
                          marginLeft: '8px' 
                        }} 
                        onClick={() => copyToClipboard(party.address)}
                        title="Copy address"
                      >
                        <Copy size={14} />
                      </button>
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      {party.transactions} transactions
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ 
                      fontWeight: 600, 
                      color: party.avgRisk > 70 ? 'var(--accent-red)' : 
                             party.avgRisk > 40 ? 'var(--accent-orange)' : 
                             'var(--accent-green)' 
                    }}>
                      {party.avgRisk.toFixed(1)}
                    </div>
                    <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                      Risk Score
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <p className="empty-state-message">No counterparty data available</p>
            )}
          </div>
        </section>
      </div>

      {/* Transaction List */}
      <section className="card">
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          marginBottom: '24px',
          flexWrap: 'wrap',
          gap: '12px'
        }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: 0 }}>
            <Activity size={20} /> Recent Transactions
          </h3>
          
          <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
            <select 
              className="input-group" 
              style={{ padding: '8px 12px', width: 'auto' }} 
              value={riskFilter} 
              onChange={(e) => setRiskFilter(e.target.value)}
            >
              <option value="all">All Risk Levels</option>
              <option value="low">Low Risk</option>
              <option value="medium">Medium Risk</option>
              <option value="high">High Risk</option>
              <option value="critical">Critical Risk</option>
            </select>
          </div>
        </div>

        <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
          {loading ? (
            <div className="loading">
              <div className="spinner" />
              <span>Loading transactions...</span>
            </div>
          ) : filteredTransactions.length > 0 ? (
            filteredTransactions.slice(0, 50).map((tx) => (
              <div key={tx.id} className="transaction-item">
                <div className="transaction-info">
                  <div className="transaction-hash">
                    {tx.hash ? `${tx.hash.slice(0, 12)}...${tx.hash.slice(-8)}` : 'Exchange Transaction'}
                    {tx.hash && (
                      <button 
                        style={{ 
                          background: 'none', 
                          border: 'none', 
                          color: 'var(--text-muted)', 
                          cursor: 'pointer', 
                          marginLeft: '8px' 
                        }} 
                        onClick={() => copyToClipboard(tx.hash)}
                        title="Copy hash"
                      >
                        <Copy size={14} />
                      </button>
                    )}
                  </div>
                  <div className="transaction-amount">
                    {tx.amount ? tx.amount.toFixed(4) : '0.0000'} {tx.asset || 'ETH'}
                    <span style={{ 
                      fontSize: '12px', 
                      color: 'var(--text-muted)', 
                      marginLeft: '12px', 
                      textTransform: 'capitalize' 
                    }}>
                      {tx.transaction_type || 'unknown'} • {tx.source || 'unknown'}
                    </span>
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                    {new Date(tx.timestamp).toLocaleString()}
                    {tx.fee && ` • Fee: ${tx.fee.toFixed(6)} ETH`}
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <div>
                    <div className={`risk-badge ${tx.risk_level || 'low'}`}>
                      {tx.risk_level || 'low'}
                    </div>
                    <div style={{ 
                      fontSize: '12px', 
                      color: 'var(--text-muted)', 
                      marginTop: '2px',
                      textAlign: 'center'
                    }}>
                      Score: {tx.risk_score ? tx.risk_score.toFixed(1) : '0.0'}
                    </div>
                  </div>
                  {tx.anomaly_detected && (
                    <AlertTriangle 
                      size={16} 
                      style={{ color: 'var(--accent-orange)' }} 
                      title="Anomaly detected" 
                    />
                  )}
                </div>
              </div>
            ))
          ) : (
            <p className="empty-state-message">
              {selectedWallet === 'all' 
                ? 'No transactions found. Start an audit to analyze blockchain data.' 
                : 'No transactions found for this wallet.'}
            </p>
          )}
        </div>
      </section>
    </>
  )
}

export default Dashboard