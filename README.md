# 🧠 CryptoAuditAI  
### AI-Native Crypto Auditing & Compliance Platform  

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

> **Auditing crypto wallets just got easier.**  
> CryptoAuditAI is an AI-driven blockchain auditing platform that analyzes wallet activity, detects risks, generates compliance reports, and provides real-time alerts — all through a modern, AI-assisted dashboard.

---

## 🚀 Overview

CryptoAuditAI unifies **AI analytics**, **on-chain data intelligence**, and **regulatory compliance automation** to make crypto wallet auditing seamless and intelligent.  

It combines blockchain ingestion, risk scoring, anomaly detection, and AI-driven reporting into a single system.

### ✨ Core Capabilities
- 🧩 **Wallet Intelligence** – Real-time ingestion of Ethereum wallet data. 
- 💬 **Conversational AI** – Ask audit, compliance, or risk questions interactively  
- ⚙️ **Risk Scoring Engine** – Programmatic & AI-based risk classification  
- 🧠 **Pattern Detection** – Detects structuring, velocity, and unusual behaviors  
- 📊 **Compliance Reporting** – Generates AI audit reports (PDF & JSON)  
- 🔔 **Live Alerts** – Auto-detects suspicious or abnormal transactions   

---

## 🧩 Project Architecture

CryptoAuditAI/
│
├── apps/
│ ├── backend/ # FastAPI backend with blockchain + AI pipelines
│ └── frontend/ # React (Vite) web dashboard interface
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Excluded files

---

## ⚙️ Backend (FastAPI)

**Location:** `apps/backend`  
**Purpose:** Handles API endpoints, data ingestion, AI processing, and report generation.

### 🧠 Features
- RESTful FastAPI backend with async endpoints  
- Integrates **Web3**, **FAISS**, and **Ollama AI**  
- Modular AI, database, and ingestion architecture  
- Automated **PDF/JSON** audit report generation  

### 🔧 Setup

### 1. Create virtual environment
cd apps/backend
python -m venv .venv
.\.venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Configure environment
Create .env (see .env.example):
DATABASE_URL=sqlite:///test.db
INFURA_API_KEY=your_infura_key
OLLAMA_MODEL=Mistral 7B

4. Run the backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

5. Check health
curl http://localhost:8000/health

💻 Frontend (React + Vite)
Location: apps/frontend
Purpose: Web dashboard for visualizing audits, transactions, and alerts.

⚙️ Setup
1. Install dependencies
cd apps/frontend
npm install

2. Run development server
npm run dev

3. Open in browser
👉 http://localhost:5173

🔍 Key Endpoints
Endpoint	Method	Description
/health	GET	Check backend status
/users/{id}/wallets	POST	Add a crypto wallet
/ingest	POST	Fetch blockchain transactions
/users/{id}/transactions	GET	View transactions with risk scores
/audit	POST	Run transaction pattern audit
/report	POST	Generate AI-powered compliance report
/reports/{id}/download	GET	Download reports (PDF/JSON)
/chat	POST	Conversational AI audit assistant

📄 Reports
Each report contains:
Risk score breakdown
High-risk transaction summary
AML/KYC compliance highlights
AI-generated narrative insights

Example outputs
audit_report.json – structured risk data

audit_report.pdf – formatted AI-written report

🧠 AI & Analytics
Component	Function
Ollama AI Model	Generates natural-language summaries & responses
FAISS Indexing	Enables fast transaction clustering & pattern search
Programmatic Risk Scoring	Real-time scoring without AI delays

🧰 Developer Utilities
Full Workflow Test (PowerShell)
You can test the complete workflow via:
apps/backend/test_complete_rag.ps1

Run it with:
.\test_complete_rag.ps1
This validates ingestion → audit → alerts → report → chat.

🔒 Security & Compliance
Follows OWASP API Security principles
.env and private data excluded via .gitignore
Sanitized wallet data handling
Read-only blockchain access (no private keys needed)

📦 Tech Stack
Layer	Technology
Backend	FastAPI (Python 3.10+), SQLite/PostgreSQL
Frontend	React (Vite + Tailwind)
AI Engine	Ollama / Mistral 7B for reporting and chat
Data Layer	Web3.py + FAISS
Infra	Uvicorn, Pydantic, Requests

🧪 Testing
Run backend tests
pytest
Run backend validation
powershell
.\apps\backend\test_backend.ps1

🧭 Roadmap
 Add multi-wallet user management

 Support Bitcoin & Solana blockchains

 Integrate live market data & exchange tagging

 Add AML compliance templates

 Enable containerized AI models for local inference

👥 Contributors
Name	Role
Zeal Thomas	Lead Developer / Architect
CryptoAuditAI Team	Backend, Frontend & AI Engineers

📜 License
This project is licensed under the Apache License 2.0.
See the LICENSE file for details.

🌐 Repository
GitHub: https://github.com/Zealthomas/CryptoAuditAI

💬 Support the Project
If you find CryptoAuditAI useful, please ⭐ the repository and share it with others in the crypto compliance and security space.
