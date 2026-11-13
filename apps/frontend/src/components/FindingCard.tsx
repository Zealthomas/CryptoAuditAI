import React, { useState } from 'react';

type FindingCardProps = {
  title: string;
  severity: 'Low' | 'Medium' | 'High' | 'Critical';
  evidence: string;
  explanation: string;
};

const severityColors = {
  Low: '#22c55e',
  Medium: '#f59e0b',
  High: '#ef4444',
  Critical: '#7f1d1d',
};

export default function FindingCard({ title, severity, evidence, explanation }: FindingCardProps) {
  const [open, setOpen] = useState(false);

  return (
    <div style={{ border: '1px solid rgba(255,255,255,0.1)', borderRadius: 12, padding: 16, background: '#1f2937', marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <strong>{title}</strong>
        <span style={{ background: severityColors[severity], color: '#fff', fontSize: 12, borderRadius: 999, padding: '4px 10px', fontWeight: 700 }}>
          {severity}
        </span>
      </div>

      <button type="button" onClick={() => setOpen((value) => !value)} style={{ marginTop: 12, background: 'transparent', color: '#93c5fd', border: 'none', cursor: 'pointer' }}>
        {open ? 'Hide evidence' : 'Show evidence'}
      </button>

      {open && (
        <div style={{ marginTop: 12, color: '#d1d5db', fontSize: 14 }}>
          <div style={{ marginBottom: 8 }}><strong>Evidence:</strong> {evidence}</div>
          <div><strong>Explanation:</strong> {explanation}</div>
        </div>
      )}
    </div>
  );
}
