// src/components/MixerInfo.jsx
import React, { useState, useEffect } from 'react'
import { useApi } from '../contexts/ApiContext'
import { Blend, Shield, AlertOctagon, Database, CheckCircle } from 'lucide-react'

const MixerInfo = () => {
  const [mixerStats, setMixerStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const { apiCall } = useApi()

  useEffect(() => {
    const loadMixerStats = async () => {
      try {
        setLoading(true)
        const stats = await apiCall('/mixer-database/stats')
        setMixerStats(stats)
      } catch (error) {
        console.error('Failed to load mixer stats:', error)
      } finally {
        setLoading(false)
      }
    }

    loadMixerStats()
  }, [apiCall])

  if (loading) {
    return (
      <section className="card">
        <div className="loading">
          <div className="spinner" />
          <span>Loading mixer database info...</span>
        </div>
      </section>
    )
  }

  if (!mixerStats) {
    return null
  }

  return (
    <section className="card">
      <h3 style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '12px', 
        marginBottom: '24px' 
      }}>
        <Blend size={24} />
        Mixer Detection Database
      </h3>

      {/* Overview Stats Grid */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
        gap: '20px',
        marginBottom: '30px'
      }}>
        <div style={{
          padding: '20px',
          background: 'linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 212, 255, 0.05) 100%)',
          border: '1px solid rgba(0, 212, 255, 0.3)',
          borderRadius: '8px'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px', 
            marginBottom: '12px',
            color: 'var(--accent-blue)'
          }}>
            <Database size={16} />
            <span style={{ fontSize: '12px', fontWeight: '600' }}>Total Mixers</span>
          </div>
          <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--accent-blue)' }}>
            {mixerStats.total_mixers}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            Tracked mixing services
          </div>
        </div>

        <div style={{
          padding: '20px',
          background: 'linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%)',
          border: '1px solid rgba(0, 255, 136, 0.3)',
          borderRadius: '8px'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px', 
            marginBottom: '12px',
            color: 'var(--accent-green)'
          }}>
            <CheckCircle size={16} />
            <span style={{ fontSize: '12px', fontWeight: '600' }}>Addresses</span>
          </div>
          <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--accent-green)' }}>
            {mixerStats.total_addresses}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            Known mixer addresses
          </div>
        </div>

        <div style={{
          padding: '20px',
          background: 'linear-gradient(135deg, rgba(255, 71, 87, 0.2) 0%, rgba(255, 71, 87, 0.05) 100%)',
          border: '2px solid var(--accent-red)',
          borderRadius: '8px'
        }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px', 
            marginBottom: '12px',
            color: 'var(--accent-red)'
          }}>
            <AlertOctagon size={16} />
            <span style={{ fontSize: '12px', fontWeight: '600' }}>OFAC Sanctioned</span>
          </div>
          <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--accent-red)' }}>
            {mixerStats.sanctioned_mixers}
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            US Treasury sanctioned
          </div>
        </div>
      </div>

      {/* Blockchain Coverage */}
      <div style={{ marginBottom: '30px' }}>
        <h4 style={{ 
          fontSize: '14px', 
          fontWeight: '600', 
          color: 'var(--text-secondary)', 
          marginBottom: '12px' 
        }}>
          Blockchain Coverage
        </h4>
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          {mixerStats.blockchains && Object.entries(mixerStats.blockchains).map(([chain, count]) => (
            <div 
              key={chain}
              style={{
                padding: '8px 16px',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                borderRadius: '6px',
                fontSize: '13px'
              }}
            >
              <span style={{ textTransform: 'capitalize', fontWeight: '600' }}>{chain}</span>
              <span style={{ color: 'var(--text-muted)', marginLeft: '8px' }}>({count})</span>
            </div>
          ))}
        </div>
      </div>

      {/* Tracked Mixers List */}
      <div>
        <h4 style={{ 
          fontSize: '14px', 
          fontWeight: '600', 
          color: 'var(--text-secondary)', 
          marginBottom: '12px' 
        }}>
          Tracked Mixing Services
        </h4>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
          gap: '12px'
        }}>
          {mixerStats.mixer_names && mixerStats.mixer_names.map((name, index) => {
            const isSanctioned = [
              'Tornado Cash', 
              'Blender.io', 
              'Sinbad'
            ].includes(name)

            return (
              <div 
                key={index}
                style={{
                  padding: '12px',
                  background: isSanctioned 
                    ? 'linear-gradient(135deg, rgba(255, 71, 87, 0.15) 0%, rgba(255, 71, 87, 0.05) 100%)'
                    : 'var(--bg-secondary)',
                  border: isSanctioned 
                    ? '1px solid var(--accent-red)' 
                    : '1px solid var(--border)',
                  borderRadius: '6px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  fontSize: '13px'
                }}
              >
                {isSanctioned ? (
                  <AlertOctagon size={14} style={{ color: 'var(--accent-red)', flexShrink: 0 }} />
                ) : (
                  <Blend size={14} style={{ color: 'var(--accent-blue)', flexShrink: 0 }} />
                )}
                <span style={{ fontWeight: '500' }}>{name}</span>
                {isSanctioned && (
                  <span style={{
                    marginLeft: 'auto',
                    fontSize: '10px',
                    padding: '2px 6px',
                    background: 'var(--accent-red)',
                    color: 'white',
                    borderRadius: '3px',
                    fontWeight: '600'
                  }}>
                    SANCTIONED
                  </span>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Info Footer */}
      <div style={{
        marginTop: '24px',
        padding: '16px',
        background: 'rgba(0, 212, 255, 0.05)',
        border: '1px solid rgba(0, 212, 255, 0.2)',
        borderRadius: '8px',
        fontSize: '12px',
        color: 'var(--text-muted)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <Shield size={14} style={{ color: 'var(--accent-blue)' }} />
          <strong style={{ color: 'var(--accent-blue)' }}>Detection Method:</strong>
        </div>
        <p style={{ margin: 0, lineHeight: '1.6' }}>
          Our system performs real-time address-based detection against a comprehensive database of known 
          cryptocurrency mixing services. Every transaction is automatically scanned for interactions with 
          these addresses, including both sending to and receiving from mixers. OFAC sanctioned entities 
          are flagged with critical severity for immediate compliance review.
        </p>
      </div>
    </section>
  )
}

export default MixerInfo