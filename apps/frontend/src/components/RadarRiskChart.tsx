import React from 'react';
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
} from 'recharts';

type RadarRiskChartProps = {
  scores: {
    contract_safety: number;
    liquidity_health: number;
    holder_concentration: number;
    market_behavior: number;
    project_reputation: number;
    wallet_behavior?: number;
  };
};

export default function RadarRiskChart({ scores }: RadarRiskChartProps) {
  const data = [
    { subject: 'Contract', score: scores.contract_safety },
    { subject: 'Liquidity', score: scores.liquidity_health },
    { subject: 'Holders', score: scores.holder_concentration },
    { subject: 'Market', score: scores.market_behavior },
    { subject: 'Reputation', score: scores.project_reputation },
    { subject: 'Wallet', score: scores.wallet_behavior ?? 50 },
  ];

  return (
    <ResponsiveContainer width="100%" height={280}>
      <RadarChart data={data} outerRadius="70%">
        <PolarGrid stroke="rgba(255,255,255,0.2)" />
        <PolarAngleAxis dataKey="subject" tick={{ fill: '#fff', fontSize: 11 }} />
        <PolarRadiusAxis domain={[0, 100]} tick={false} />
        <Radar name="Risk" dataKey="score" stroke="#22c55e" fill="#22c55e" fillOpacity={0.35} />
      </RadarChart>
    </ResponsiveContainer>
  );
}
