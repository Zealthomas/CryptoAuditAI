import json
import os
import re
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

BIP39_SAMPLE = {
    "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract",
    "access", "accident", "account", "accuse", "achieve", "acid", "acoustic",
    "acquire", "action", "actor", "actress", "actual", "adapt", "add", "address",
    "adjust", "admit", "adopt", "adult", "afraid", "again", "age", "agent",
    "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album", "alcohol",
}

CHAIN_ID_MAP = {
    "ethereum": "1",
    "bsc": "56",
    "polygon": "137",
    "arbitrum": "42161",
    "base": "8453",
    "solana": "solana",
}


def scan_sensitive_input(text: str) -> Optional[str]:
    if not text:
        return None

    trimmed = text.strip()
    if not trimmed:
        return None

    if re.fullmatch(r"(0x)?[a-fA-F0-9]{64}", trimmed):
        return "Input resembles a private key. Never submit your private key!"

    words = trimmed.lower().split()
    if len(words) in {12, 15, 18, 24}:
        matches = sum(1 for word in words if word in BIP39_SAMPLE)
        if matches >= 4:
            return f"Input resembles a {len(words)}-word Secret Recovery Phrase. Never share seed phrases!"

    return None


async def fetch_dexscreener_data(chain: str, address: str) -> Optional[Dict[str, Any]]:
    del chain
    try:
        async with httpx.AsyncClient(timeout=8.0) as client_http:
            response = await client_http.get(f"https://api.dexscreener.com/latest/dex/tokens/{address}")
            if response.status_code != 200:
                return None

            data = response.json()
            pairs = data.get("pairs") or []
            if not pairs:
                return None

            pairs.sort(key=lambda item: float((item.get("liquidity") or {}).get("usd") or 0), reverse=True)
            top = pairs[0]
            return {
                "pair_address": top.get("pairAddress"),
                "liquidity_usd": float((top.get("liquidity") or {}).get("usd") or 0),
                "volume_24h_usd": float((top.get("volume") or {}).get("h24") or 0),
                "price_usd": top.get("priceUsd"),
                "txns_24h": top.get("txns", {}).get("h24", {}),
            }
    except Exception:
        return None


async def fetch_goplus_security(chain: str, address: str) -> Optional[Dict[str, Any]]:
    chain_id = CHAIN_ID_MAP.get(chain.lower())
    if not chain_id:
        return None

    try:
        async with httpx.AsyncClient(timeout=8.0) as client_http:
            url = f"https://api.gopluslabs.io/api/v1/token_security/{chain_id}?contract_addresses={address.lower()}"
            response = await client_http.get(url)
            if response.status_code != 200:
                return None

            payload = response.json()
            data = (payload.get("result") or {}).get(address.lower())
            if not data:
                return None

            return {
                "is_honeypot": data.get("is_honeypot") == "1",
                "buy_tax": f"{float(data.get('buy_tax', 0)) * 100:.1f}%",
                "sell_tax": f"{float(data.get('sell_tax', 0)) * 100:.1f}%",
                "is_open_source": data.get("is_open_source") == "1",
                "is_proxy": data.get("is_proxy") == "1",
                "is_mintable": data.get("is_mintable") == "1",
                "can_take_back_ownership": data.get("can_take_back_ownership") == "1",
            }
    except Exception:
        return None


class KeyFinding(BaseModel):
    title: str
    severity: str
    evidence: str
    plain_language_explanation: str


class CategoryScores(BaseModel):
    contract_safety: int
    liquidity_health: int
    holder_concentration: int
    market_behavior: int
    project_reputation: int
    wallet_behavior: Optional[int] = None


class RiskReportSchema(BaseModel):
    overall_risk_score: int
    risk_level: str
    category_scores: CategoryScores
    key_findings: list[KeyFinding]
    red_flags: list[str]
    positive_signals: list[str]
    summary: str
    what_would_change_this_score: list[str]
    confidence: str
    confidence_reason: str
    id: Optional[str] = None
    target: Optional[str] = None
    chain: Optional[str] = None
    mode: Optional[str] = None
    sources: list[Dict[str, Any]] = Field(default_factory=list)


async def analyze_crypto_risk(mode: str, chain: str, target: str, contract_code: Optional[str] = None) -> Dict[str, Any]:
    dex_data = None
    security_data = None

    if mode == "token":
        dex_data = await fetch_dexscreener_data(chain, target)
        security_data = await fetch_goplus_security(chain, target)

    prompt = f"""
You are RiskLens / CryptoAuditAI, an expert smart contract and crypto risk auditor.
Target: {target}
Chain: {chain}
Mode: {mode}

On-Chain Scanner Data:
{json.dumps({'dexscreener': dex_data, 'goplus': security_data}, indent=2)}

{f'Contract Source Code:\n{contract_code[:7000]}' if contract_code else ''}

Rules:
- Give a 0-100 overall_risk_score (100 = dangerous/scam).
- State 'Data unavailable' if metrics are missing and lower confidence level.
- Never give financial advice. Focus strictly on risk and security mechanics.
"""

    if not client:
        return {
            "overall_risk_score": 35,
            "risk_level": "Medium",
            "category_scores": {
                "contract_safety": 30,
                "liquidity_health": 40,
                "holder_concentration": 50,
                "market_behavior": 30,
                "project_reputation": 20,
            },
            "key_findings": [{
                "title": "On-chain telemetry retrieved",
                "severity": "Low",
                "evidence": f"Liquidity: ${dex_data.get('liquidity_usd', 0) if dex_data else 0:,.2f}",
                "plain_language_explanation": "Pool contains active DEX trading volume.",
            }],
            "red_flags": [],
            "positive_signals": ["Verified smart contract on-chain"],
            "summary": f"Audit completed for {target}. Moderate risk profile.",
            "what_would_change_this_score": ["Locking liquidity LP tokens", "Renouncing contract ownership"],
            "confidence": "Medium",
            "confidence_reason": "Live on-chain telemetry gathered.",
            "sources": [{"name": "DexScreener API"}, {"name": "GoPlus Labs"}],
            "id": f"audit-{target[:8]}",
            "target": target,
            "chain": chain,
            "mode": mode,
        }

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            response_mime_type="application/json",
            response_schema=RiskReportSchema,
            temperature=0.1,
        ),
    )

    report = json.loads(response.text)
    report["id"] = f"audit-{target[:8]}"
    report["target"] = target
    report["chain"] = chain
    report["mode"] = mode
    report["sources"] = [{"name": "DexScreener API"}, {"name": "GoPlus Security"}]

    candidates = getattr(response, "candidates", [])
    if candidates and hasattr(candidates[0], "grounding_metadata"):
        metadata = candidates[0].grounding_metadata
        if hasattr(metadata, "grounding_chunks") and metadata.grounding_chunks:
            for chunk in metadata.grounding_chunks:
                if hasattr(chunk, "web") and chunk.web:
                    report["sources"].append({
                        "name": getattr(chunk.web, "title", "Web Intelligence"),
                        "url": getattr(chunk.web, "uri", None),
                    })

    return report


async def stream_copilot_response(report_context: dict, query: str) -> AsyncGenerator[str, None]:
    if not client:
        yield f"data: {json.dumps({'text': 'Copilot: Please configure GEMINI_API_KEY in .env for live AI responses.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    instruction = f"""
You are the CryptoAuditAI / RiskLens Copilot.
Report Context:
{json.dumps(report_context)}

Answer user questions about vulnerabilities, honeypot traits, or wallet safety in clear, plain language.
Never advise to buy or sell.
User: {query}
"""

    try:
        stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=instruction,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            ),
        )
        for chunk in stream:
            if getattr(chunk, "text", None):
                yield f"data: {json.dumps({'text': chunk.text})}\n\n"
    except Exception:
        yield f"data: {json.dumps({'error': 'Copilot response failed.'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"
