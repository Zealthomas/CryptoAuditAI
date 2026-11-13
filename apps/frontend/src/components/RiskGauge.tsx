import React from 'react';

type RiskGaugeProps = {
  score: number;
};

export default function RiskGauge({ score }: RiskGaugeProps) {
  const safeScore = Math.max(0, Math.min(100, score || 0));
  const angle = -120 + (safeScore / 100) * 240;

  return (
    <svg viewBox="0 0 220 120" width="100%" height="130" aria-label="Risk gauge">
      <defs>
        <linearGradient id="riskGaugeGradient" x1="0%" x2="100%" y1="0%" y2="0%">
          <stop offset="0%" stopColor="#22c55e" />
          <stop offset="50%" stopColor="#f59e0b" />
          <stop offset="100%" stopColor="#ef4444" />
        </linearGradient>
      </defs>

      <path d="M20 90 A90 90 0 0 1 200 90" fill="none" stroke="url(#riskGaugeGradient)" strokeWidth="18" strokeLinecap="round" />
      <path d="M20 90 A90 90 0 0 1 200 90" fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth="26" strokeLinecap="round" />

      <g transform={`translate(110 90)`}>
        <line x1="0" y1="0" x2="0" y2="-65" stroke="#fff" strokeWidth="4" transform={`rotate(${angle})`} />
        <circle r="8" fill="#fff" />
      </g>

      <text x="110" y="90" textAnchor="middle" fill="#fff" fontSize="30" fontWeight="700">
        {safeScore}
      </text>
    </svg>
  );
}
