"""
Complete AI Engine for CryptoAuditAI
Real AI-powered risk assessment, anomaly detection, and compliance analysis using Mistral 7B
"""

import logging
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import httpx
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer

from database import AsyncSessionLocal
from models import Transaction, User, Alert, AuditReport
from config import settings

logger = logging.getLogger(__name__)

# Initialize embedding model for similarity analysis
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None

@dataclass
class RiskAnalysis:
    risk_score: float
    risk_level: str
    anomaly_detected: bool
    compliance_flags: List[str]
    reasoning: str

@dataclass
class TransactionPattern:
    pattern_type: str
    confidence: float
    transactions: List[int]
    description: str

class CryptoAIEngine:
    def __init__(self):
        self.ollama_url = settings.OLLAMA_HOST
        self.model = settings.OLLAMA_MODEL
        self.embedding_model = embedding_model
        
    async def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Call Ollama API for LLM inference"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 512
                    }
                }
                
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=120.0
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "").strip()
                
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return "AI analysis temporarily unavailable"
    
    def _calculate_base_risk_score(self, transaction: Transaction) -> float:
        """Calculate base risk score using programmatic rules"""
        risk_factors = []
        
        # Amount-based risk
        if transaction.amount > 100000:  # Large amounts
            risk_factors.append(30)
        elif transaction.amount > 50000:
            risk_factors.append(20)
        elif transaction.amount > 10000:
            risk_factors.append(10)
        
        # Gas price anomalies (for Ethereum)
        if transaction.gas_price and transaction.gas_price > 100:  # High gas price
            risk_factors.append(15)
        
        # Transaction status
        if transaction.status == "failed":
            risk_factors.append(25)
        
        # Time-based patterns (transactions outside business hours)
        hour = transaction.timestamp.hour
        if hour < 6 or hour > 22:  # Outside business hours
            risk_factors.append(10)
        
        # Fee anomalies
        if transaction.fee and transaction.amount > 0:
            fee_ratio = transaction.fee / transaction.amount
            if fee_ratio > 0.05:  # High fee ratio
                risk_factors.append(20)
        
        return min(sum(risk_factors), 100)
    
    def _detect_anomalies(self, transaction: Transaction, user_transactions: List[Transaction]) -> bool:
        """Detect anomalies based on user's transaction patterns"""
        if not user_transactions or len(user_transactions) < 5:
            return False
        
        # Calculate user's average transaction amount
        amounts = [t.amount for t in user_transactions if t.id != transaction.id]
        if not amounts:
            return False
            
        avg_amount = np.mean(amounts)
        std_amount = np.std(amounts)
        
        # Amount-based anomaly (more than 3 standard deviations)
        if std_amount > 0 and abs(transaction.amount - avg_amount) > 3 * std_amount:
            return True
        
        # Frequency-based anomaly (too many transactions in short time)
        recent_transactions = [
            t for t in user_transactions 
            if t.timestamp > transaction.timestamp - timedelta(hours=1)
            and t.id != transaction.id
        ]
        
        if len(recent_transactions) > 10:  # More than 10 transactions in 1 hour
            return True
        
        return False
    
    def _get_compliance_flags(self, transaction: Transaction) -> List[str]:
        """Identify compliance issues"""
        flags = []
        
        # Large transaction reporting thresholds
        if transaction.amount > 10000:
            flags.append("CTR_THRESHOLD")  # Currency Transaction Report
        
        # Suspicious activity patterns
        if transaction.amount == 9999 or transaction.amount == 9500:  # Just under reporting threshold
            flags.append("STRUCTURING_SUSPICION")
        
        # High-risk jurisdictions (would need actual address-to-country mapping)
        if transaction.to_address and len(transaction.to_address) > 0:
            # Placeholder - in real implementation, check against OFAC lists
            pass
        
        # Round number patterns (often indicating manual/suspicious transactions)
        if transaction.amount % 1000 == 0 and transaction.amount > 5000:
            flags.append("ROUND_AMOUNT_PATTERN")
        
        return flags
    
    async def analyze_transaction(self, transaction_id: int) -> RiskAnalysis:
        """Perform comprehensive AI-powered risk analysis on a single transaction"""
        async with AsyncSessionLocal() as session:
            # Get the transaction
            transaction = await session.get(Transaction, transaction_id)
            if not transaction:
                raise ValueError(f"Transaction {transaction_id} not found")
            
            # Get user's transaction history for context
            user_transactions_result = await session.execute(
                select(Transaction)
                .where(Transaction.user_id == transaction.user_id)
                .order_by(Transaction.timestamp.desc())
                .limit(100)
            )
            user_transactions = user_transactions_result.scalars().all()
            
            # Calculate base risk score
            base_risk = self._calculate_base_risk_score(transaction)
            
            # Detect anomalies
            anomaly_detected = self._detect_anomalies(transaction, user_transactions)
            
            # Get compliance flags
            compliance_flags = self._get_compliance_flags(transaction)
            
            # Prepare AI analysis prompt
            transaction_context = f"""
Transaction Details:
- Hash: {transaction.hash}
- Amount: {transaction.amount} {transaction.asset}
- Type: {transaction.transaction_type}
- From: {transaction.from_address}
- To: {transaction.to_address}
- Source: {transaction.source}
- Timestamp: {transaction.timestamp}
- Fee: {transaction.fee}
- Status: {transaction.status}

Base Risk Score: {base_risk}/100
Anomaly Detected: {anomaly_detected}
Compliance Flags: {compliance_flags}

User Transaction History Summary:
- Total transactions: {len(user_transactions)}
- Average amount: {np.mean([t.amount for t in user_transactions]):.2f}
- Primary sources: {list(set([t.source for t in user_transactions[:10]]))}
"""
            
            system_prompt = """You are an expert cryptocurrency forensic analyst. Analyze the transaction data and provide a risk assessment. Focus on:
1. Money laundering indicators
2. Suspicious patterns
3. Compliance violations
4. Market manipulation signs
5. Terrorist financing risks

Provide clear, actionable reasoning for your risk assessment."""
            
            # Get AI analysis
            ai_reasoning = await self._call_ollama(transaction_context, system_prompt)
            
            # Adjust risk score based on flags and anomalies
            final_risk = base_risk
            if anomaly_detected:
                final_risk = min(final_risk + 20, 100)
            if len(compliance_flags) > 0:
                final_risk = min(final_risk + (len(compliance_flags) * 15), 100)
            
            # Determine risk level
            if final_risk >= 80:
                risk_level = "critical"
            elif final_risk >= 60:
                risk_level = "high"
            elif final_risk >= 40:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return RiskAnalysis(
                risk_score=final_risk,
                risk_level=risk_level,
                anomaly_detected=anomaly_detected,
                compliance_flags=compliance_flags,
                reasoning=ai_reasoning
            )
    
    async def detect_suspicious_patterns(self, user_id: int, days: int = 30) -> List[TransactionPattern]:
        """Detect suspicious patterns in user's transaction history"""
        async with AsyncSessionLocal() as session:
            # Get recent transactions
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            result = await session.execute(
                select(Transaction)
                .where(Transaction.user_id == user_id)
                .where(Transaction.timestamp >= cutoff_date)
                .order_by(Transaction.timestamp.desc())
            )
            transactions = result.scalars().all()
            
            patterns = []
            
            if len(transactions) < 3:
                return patterns
            
            # Pattern 1: Rapid succession transactions
            rapid_transactions = []
            for i in range(len(transactions) - 1):
                time_diff = abs((transactions[i].timestamp - transactions[i+1].timestamp).total_seconds())
                if time_diff < 300:  # Less than 5 minutes apart
                    rapid_transactions.extend([transactions[i].id, transactions[i+1].id])
            
            if len(rapid_transactions) >= 4:
                patterns.append(TransactionPattern(
                    pattern_type="rapid_succession",
                    confidence=0.8,
                    transactions=list(set(rapid_transactions)),
                    description="Multiple transactions executed within minutes of each other"
                ))
            
            # Pattern 2: Round number structuring
            round_amounts = [t for t in transactions if t.amount % 1000 == 0 and t.amount >= 5000]
            if len(round_amounts) >= 3:
                patterns.append(TransactionPattern(
                    pattern_type="round_amount_structuring",
                    confidence=0.7,
                    transactions=[t.id for t in round_amounts],
                    description="Pattern of round-number transactions suggesting manual intervention"
                ))
            
            # Pattern 3: Just-under-threshold transactions
            threshold_avoidance = [t for t in transactions if 9000 <= t.amount <= 9999]
            if len(threshold_avoidance) >= 2:
                patterns.append(TransactionPattern(
                    pattern_type="threshold_avoidance",
                    confidence=0.9,
                    transactions=[t.id for t in threshold_avoidance],
                    description="Transactions just under reporting thresholds (potential structuring)"
                ))
            
            return patterns
    
    async def generate_compliance_assessment(self, user_id: int) -> Dict[str, Any]:
        """Generate AI-powered compliance assessment"""
        async with AsyncSessionLocal() as session:
            # Get user transactions
            result = await session.execute(
                select(Transaction).where(Transaction.user_id == user_id)
            )
            transactions = result.scalars().all()
            
            if not transactions:
                return {
                    "compliance_score": 100,
                    "risk_level": "low",
                    "findings": [],
                    "recommendations": []
                }
            
            # Analyze compliance issues
            high_risk_count = len([t for t in transactions if t.risk_level in ["high", "critical"]])
            total_flagged = len([t for t in transactions if t.compliance_flags])
            large_transactions = len([t for t in transactions if t.amount > 10000])
            
            # Calculate compliance score
            compliance_score = max(0, 100 - (high_risk_count * 10) - (total_flagged * 5))
            
            findings = []
            recommendations = []
            
            if high_risk_count > 0:
                findings.append(f"{high_risk_count} high-risk transactions identified")
                recommendations.append("Review high-risk transactions and document business justification")
            
            if large_transactions > 0:
                findings.append(f"{large_transactions} transactions above $10,000 reporting threshold")
                recommendations.append("Ensure CTR (Currency Transaction Report) compliance")
            
            # Prepare AI prompt for detailed analysis
            compliance_context = f"""
Compliance Analysis Request:
- Total transactions: {len(transactions)}
- High-risk transactions: {high_risk_count}
- Flagged transactions: {total_flagged}
- Large transactions (>$10k): {large_transactions}
- Compliance score: {compliance_score}/100

Key findings: {findings}
"""
            
            system_prompt = """You are a compliance officer specializing in AML/KYC regulations. Provide specific recommendations for improving compliance posture and managing regulatory risk."""
            
            ai_recommendations = await self._call_ollama(compliance_context, system_prompt)
            
            return {
                "compliance_score": compliance_score,
                "risk_level": "high" if compliance_score < 60 else "medium" if compliance_score < 80 else "low",
                "findings": findings,
                "ai_recommendations": ai_recommendations,
                "recommendations": recommendations,
                "total_transactions": len(transactions),
                "flagged_count": total_flagged
            }
    
    async def create_alert(self, user_id: int, transaction_id: Optional[int], 
                          title: str, message: str, severity: str, alert_type: str):
        """Create a new alert for the user"""
        async with AsyncSessionLocal() as session:
            alert = Alert(
                user_id=user_id,
                transaction_id=transaction_id,
                title=title,
                message=message,
                severity=severity,
                alert_type=alert_type
            )
            session.add(alert)
            await session.commit()
            logger.info(f"Created {severity} alert for user {user_id}: {title}")

# Global AI engine instance
ai_engine = CryptoAIEngine()

# Convenience functions for main.py
async def analyze_transaction_risk(transaction_id: int) -> RiskAnalysis:
    """Analyze a single transaction's risk"""
    return await ai_engine.analyze_transaction(transaction_id)

async def detect_user_patterns(user_id: int) -> List[TransactionPattern]:
    """Detect suspicious patterns for a user"""
    return await ai_engine.detect_suspicious_patterns(user_id)

async def get_compliance_assessment(user_id: int) -> Dict[str, Any]:
    """Get comprehensive compliance assessment"""
    return await ai_engine.generate_compliance_assessment(user_id)

async def create_risk_alert(user_id: int, transaction_id: int, risk_analysis: RiskAnalysis):
    """Create alerts based on risk analysis"""
    if risk_analysis.risk_level in ["high", "critical"]:
        await ai_engine.create_alert(
            user_id=user_id,
            transaction_id=transaction_id,
            title=f"{risk_analysis.risk_level.title()} Risk Transaction Detected",
            message=f"Transaction flagged with {risk_analysis.risk_score:.1f}/100 risk score. Compliance flags: {', '.join(risk_analysis.compliance_flags)}",
            severity=risk_analysis.risk_level,
            alert_type="risk"
        )
    
    if risk_analysis.anomaly_detected:
        await ai_engine.create_alert(
            user_id=user_id,
            transaction_id=transaction_id,
            title="Anomalous Transaction Pattern",
            message="Transaction deviates significantly from user's normal patterns",
            severity="warning",
            alert_type="anomaly"
        )