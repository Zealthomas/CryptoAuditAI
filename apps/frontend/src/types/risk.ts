export type AnalysisMode = 'token' | 'wallet' | 'contract';
export type SupportedChain = 'ethereum' | 'bsc' | 'base' | 'polygon' | 'arbitrum' | 'solana';
export type RiskSeverity = 'Low' | 'Medium' | 'High' | 'Critical';
export type ConfidenceLevel = 'Low' | 'Medium' | 'High';

export interface CategoryScores {
  contract_safety: number;
  liquidity_health: number;
  holder_concentration: number;
  market_behavior: number;
  project_reputation: number;
  wallet_behavior?: number;
}

export interface KeyFinding {
  title: string;
  severity: RiskSeverity;
  evidence: string;
  plain_language_explanation: string;
}

export interface SourceCitation {
  name: string;
  url?: string;
  details?: string;
}

export interface RiskReport {
  id: string;
  target: string;
  mode: AnalysisMode;
  chain: SupportedChain;
  timestamp: string;
  overall_risk_score: number;
  risk_level: RiskSeverity;
  category_scores: CategoryScores;
  key_findings: KeyFinding[];
  red_flags: string[];
  positive_signals: string[];
  summary: string;
  what_would_change_this_score: string[];
  confidence: ConfidenceLevel;
  confidence_reason: string;
  sources: SourceCitation[];
}

export interface WatchlistItem {
  id: string;
  target: string;
  chain: SupportedChain;
  mode: AnalysisMode;
  riskScore: number;
  previousScore?: number;
  riskLevel: RiskSeverity;
  updatedAt: string;
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
}
