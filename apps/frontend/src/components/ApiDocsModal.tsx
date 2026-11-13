import React from 'react';

type Props = {
  open: boolean;
  onClose: () => void;
};

export default function ApiDocsModal({ open, onClose }: Props) {
  if (!open) return null;

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000 }}>
      <div style={{ width: 720, maxWidth: '90vw', background: '#111827', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 16, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <h3 style={{ margin: 0 }}>RiskLens API</h3>
          <button type="button" onClick={onClose} style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer' }}>Close</button>
        </div>

        <div style={{ marginBottom: 12 }}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Risk audit endpoint</div>
          <pre style={{ whiteSpace: 'pre-wrap', background: '#0f172a', padding: 12, borderRadius: 8, color: '#e2e8f0' }}>
{`curl -X POST http://localhost:8000/api/v1/risk \\
  -H "Content-Type: application/json" \\
  -d '{"chain":"ethereum","address":"0x1234...","mode":"token"}'`}
          </pre>
        </div>

        <div>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Streaming chat endpoint</div>
          <pre style={{ whiteSpace: 'pre-wrap', background: '#0f172a', padding: 12, borderRadius: 8, color: '#e2e8f0' }}>
{`curl -N -X POST http://localhost:8000/api/chat/stream \\
  -H "Content-Type: application/json" \\
  -d '{"reportContext":{"target":"0x1234..."},"query":"Explain the risk profile"}'`}
          </pre>
        </div>
      </div>
    </div>
  );
}
