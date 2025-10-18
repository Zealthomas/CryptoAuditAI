"""
Complete Reports Engine for CryptoAuditAI
AI-powered report generation with PDF export capabilities
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
import httpx
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import blue, black

from database import AsyncSessionLocal
from models import Transaction, AuditReport, Alert
from config import settings
from ai_engine import get_compliance_assessment

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        self.reports_dir = Path(settings.REPORTS_DIR)
        self.reports_dir.mkdir(exist_ok=True)
        self.ollama_url = settings.OLLAMA_HOST
        self.model = settings.OLLAMA_MODEL

    async def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Call Ollama for report generation"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "num_predict": 2048
                    }
                }
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=180.0
                )
                response.raise_for_status()
                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return "AI report generation temporarily unavailable"

    async def _gather_user_stats(
        self, user_id: int, date_from: Optional[datetime] = None, 
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Gather comprehensive user statistics with extended risk scoring"""
        async with AsyncSessionLocal() as session:
            query_filters = [Transaction.user_id == user_id]
            if date_from:
                query_filters.append(Transaction.timestamp >= date_from)
            if date_to:
                query_filters.append(Transaction.timestamp <= date_to)

            transactions_result = await session.execute(
                select(Transaction).where(and_(*query_filters))
            )
            transactions = transactions_result.scalars().all()

            alerts_result = await session.execute(
                select(Alert).where(Alert.user_id == user_id).order_by(Alert.created_at.desc())
            )
            alerts = alerts_result.scalars().all()

            total_transactions = len(transactions)
            total_volume = sum(t.amount for t in transactions)
            total_fees = sum(t.fee or 0 for t in transactions)

            risk_weights = {"low": 1, "medium": 3, "high": 7, "critical": 10}
            risk_scores = [risk_weights.get(t.risk_level, 0) for t in transactions]
            total_risk_score = sum(risk_scores)
            avg_risk_score = total_risk_score / max(total_transactions, 1)

            high_risk_count = len([t for t in transactions if t.risk_level in ["high", "critical"]])
            medium_risk_count = len([t for t in transactions if t.risk_level == "medium"])
            anomalous_count = len([t for t in transactions if t.anomaly_detected])

            sources = {}
            assets = {}
            hourly_distribution = {}
            for tx in transactions:
                sources[tx.source] = sources.get(tx.source, 0) + 1
                assets[tx.asset] = assets.get(tx.asset, 0) + 1
                hour = tx.timestamp.hour
                hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1

            critical_alerts = len([a for a in alerts if a.severity == "critical"])
            unresolved_alerts = len([a for a in alerts if not a.is_resolved])

            return {
                "total_transactions": total_transactions,
                "total_volume": total_volume,
                "total_fees": total_fees,
                "average_transaction": total_volume / max(total_transactions, 1),
                "high_risk_count": high_risk_count,
                "medium_risk_count": medium_risk_count,
                "anomalous_count": anomalous_count,
                "critical_alerts": critical_alerts,
                "unresolved_alerts": unresolved_alerts,
                "risk_percentage": (high_risk_count / max(total_transactions, 1)) * 100,
                "total_risk_score": total_risk_score,
                "average_risk_score": avg_risk_score,
                "sources": sources,
                "assets": assets,
                "hourly_distribution": hourly_distribution,
                "date_range": {
                    "from": date_from.isoformat() if date_from else None,
                    "to": date_to.isoformat() if date_to else None
                }
            }

    async def generate_executive_summary(self, stats: Dict[str, Any], compliance: Dict[str, Any]) -> str:
        """Generate AI-powered executive summary"""
        context = f"""
User Transaction Analysis Summary:
- Total Transactions: {stats['total_transactions']}
- Total Volume: ${stats['total_volume']:,.2f}
- Average Transaction: ${stats['average_transaction']:,.2f}
- High Risk Transactions: {stats['high_risk_count']} ({stats['risk_percentage']:.1f}%)
- Anomalous Transactions: {stats['anomalous_count']}
- Critical Alerts: {stats['critical_alerts']}
- Unresolved Alerts: {stats['unresolved_alerts']}

Compliance Assessment:
- Compliance Score: {compliance['compliance_score']}/100
- Risk Level: {compliance['risk_level']}
- Key Findings: {compliance.get('findings', [])}

Top Trading Sources: {list(stats['sources'].keys())[:3]}
Primary Assets: {list(stats['assets'].keys())[:3]}
"""
        system_prompt = """You are a senior financial compliance officer writing an executive summary for a cryptocurrency audit report. 
Provide a concise but comprehensive summary that covers:
1. Overall risk assessment
2. Key findings and concerns
3. Compliance status
4. Immediate action items
Write in professional, clear language suitable for executives and regulators."""
        return await self._call_ollama(context, system_prompt)

    async def generate_detailed_analysis(self, user_id: int, stats: Dict[str, Any]) -> str:
        """Generate detailed AI analysis of transaction patterns"""
        async with AsyncSessionLocal() as session:
            high_risk_result = await session.execute(
                select(Transaction)
                .where(Transaction.user_id == user_id)
                .where(Transaction.risk_level.in_(["high", "critical"]))
                .limit(10)
            )
            high_risk_transactions = high_risk_result.scalars().all()

            high_risk_details = [
                f"- ${tx.amount} {tx.asset} on {tx.timestamp.strftime('%Y-%m-%d')} "
                f"(Risk: {tx.risk_level}, Source: {tx.source})"
                for tx in high_risk_transactions
            ]

            context = f"""
Detailed Transaction Pattern Analysis:

High-Risk Transactions ({len(high_risk_transactions)} total):
{chr(10).join(high_risk_details[:5])}

Statistical Patterns:
- Most active trading hours: {sorted(stats['hourly_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]}
- Primary sources: {stats['sources']}
- Asset distribution: {stats['assets']}
- Risk distribution: {stats['high_risk_count']} high, {stats['medium_risk_count']} medium risk

Anomaly Analysis:
- {stats['anomalous_count']} transactions flagged as anomalous
- Risk percentage: {stats['risk_percentage']:.1f}%
"""
            system_prompt = """You are a blockchain forensics expert analyzing cryptocurrency transaction patterns. 
Provide detailed analysis covering:
1. Pattern recognition and behavioral analysis
2. Risk factor identification
3. Potential money laundering indicators
4. Regulatory compliance concerns
5. Specific recommendations for risk mitigation
Be specific and technical while remaining actionable."""
            return await self._call_ollama(context, system_prompt)

    def _create_pdf_report(self, report_data: Dict[str, Any], filepath: str):
        """Create PDF report using ReportLab"""
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=blue,
            alignment=1
        )
        story.append(Paragraph("CryptoAuditAI Security Report", title_style))
        story.append(Spacer(1, 20))

        # Metadata
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>Report Type:</b> {report_data.get('report_type', 'Comprehensive')}", styles['Normal']))
        story.append(Paragraph(f"<b>User ID:</b> {report_data.get('user_id')}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading1']))
        story.append(Paragraph(report_data.get('summary', ''), styles['Normal']))
        story.append(Spacer(1, 15))

        # Key Statistics Table
        story.append(Paragraph("Key Statistics", styles['Heading2']))
        stats_data = [
            ['Metric', 'Value'],
            ['Total Transactions', str(report_data['stats']['total_transactions'])],
            ['Total Volume', f"${report_data['stats']['total_volume']:,.2f}"],
            ['High Risk Transactions', str(report_data['stats']['high_risk_count'])],
            ['Risk Percentage', f"{report_data['stats']['risk_percentage']:.1f}%"],
            ['Total Risk Score', str(report_data['stats']['total_risk_score'])],
            ['Average Risk Score', f"{report_data['stats']['average_risk_score']:.2f}"],
            ['Compliance Score', f"{report_data['compliance']['compliance_score']}/100"],
            ['Critical Alerts', str(report_data['stats']['critical_alerts'])]
        ]
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), '#f0f0f0'),
            ('GRID', (0, 0), (-1, -1), 1, black)
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 20))

        # Detailed Analysis
        story.append(Paragraph("Detailed Analysis", styles['Heading2']))
        story.append(Paragraph(report_data.get('detailed_analysis', ''), styles['Normal']))
        story.append(Spacer(1, 15))

        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        recommendations = report_data.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        story.append(Spacer(1, 15))

        # Compliance Assessment
        story.append(Paragraph("Compliance Assessment", styles['Heading2']))
        compliance_text = f"""
        <b>Overall Score:</b> {report_data['compliance']['compliance_score']}/100<br/>
        <b>Risk Level:</b> {report_data['compliance']['risk_level'].title()}<br/>
        <b>Key Findings:</b><br/>
        """
        for finding in report_data['compliance'].get('findings', []):
            compliance_text += f"• {finding}<br/>"
        story.append(Paragraph(compliance_text, styles['Normal']))

        doc.build(story)

    async def generate_comprehensive_report(
        self, user_id: int, report_type: str = "comprehensive",
        date_from: Optional[datetime] = None, date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive audit report"""
        stats = await self._gather_user_stats(user_id, date_from, date_to)
        compliance = await get_compliance_assessment(user_id)

        summary = await self.generate_executive_summary(stats, compliance)
        detailed_analysis = await self.generate_detailed_analysis(user_id, stats)

        recommendations_context = f"""
        Based on the analysis:
        - Risk Level: {compliance['risk_level']}
        - High Risk Transactions: {stats['high_risk_count']}
        - Compliance Score: {compliance['compliance_score']}/100
        - Unresolved Alerts: {stats['unresolved_alerts']}
        """
        recommendations_prompt = "Provide 5 specific, actionable recommendations to improve security and compliance posture."
        ai_recommendations = await self._call_ollama(recommendations_context, recommendations_prompt)

        recommendations = [
            line.lstrip('0123456789.-• ').strip()
            for line in ai_recommendations.split('\n')
            if line.strip() and (line[0].isdigit() or line.startswith('-') or line.startswith('•'))
        ][:5]

        report_data = {
            "user_id": user_id,
            "report_type": report_type,
            "summary": summary,
            "detailed_analysis": detailed_analysis,
            "stats": stats,
            "compliance": compliance,
            "recommendations": recommendations,
            "generated_at": datetime.now()
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filepath = self.reports_dir / f"audit_report_{user_id}_{timestamp}.json"
        pdf_filepath = self.reports_dir / f"audit_report_{user_id}_{timestamp}.pdf"

        with open(json_filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        self._create_pdf_report(report_data, str(pdf_filepath))

        async with AsyncSessionLocal() as session:
            db_report = AuditReport(
                user_id=user_id,
                title=f"{report_type.title()} Audit Report",
                report_type=report_type,
                summary=summary[:1000],
                full_report=detailed_analysis,
                recommendations=recommendations,
                overall_risk_score=stats['risk_percentage'],
                total_risk_score=stats['total_risk_score'],
                average_risk_score=stats['average_risk_score'],
                total_transactions=stats['total_transactions'],
                flagged_transactions=stats['high_risk_count'],
                compliance_score=compliance['compliance_score'],
                pdf_path=str(pdf_filepath),
                json_path=str(json_filepath)
            )
            session.add(db_report)
            await session.commit()
            await session.refresh(db_report)

            logger.info(f"Generated comprehensive report for user {user_id}: {pdf_filepath.name}")

        return {
            "report_id": db_report.id,
            "pdf_path": str(pdf_filepath),
            "json_path": str(json_filepath),
            "summary": summary,
            "overall_risk_score": stats['risk_percentage'],
            "compliance_score": compliance['compliance_score'],
            "total_transactions": stats['total_transactions'],
            "flagged_transactions": stats['high_risk_count']
        }

# Global report generator instance
report_generator = ReportGenerator()

async def generate_audit_report(user_id: int, report_type: str = "comprehensive",
                                date_from: Optional[datetime] = None,
                                date_to: Optional[datetime] = None) -> Dict[str, Any]:
    """Generate audit report for user"""
    return await report_generator.generate_comprehensive_report(user_id, report_type, date_from, date_to)

async def get_user_reports(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Get user's recent audit reports"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(AuditReport)
            .where(AuditReport.user_id == user_id)
            .order_by(AuditReport.created_at.desc())
            .limit(limit)
        )
        reports = result.scalars().all()

        return [{
            "id": report.id,
            "title": report.title,
            "report_type": report.report_type,
            "overall_risk_score": report.overall_risk_score,
            "compliance_score": report.compliance_score,
            "total_transactions": report.total_transactions,
            "flagged_transactions": report.flagged_transactions,
            "created_at": report.created_at,
            "pdf_path": report.pdf_path,
            "json_path": report.json_path
        } for report in reports]
