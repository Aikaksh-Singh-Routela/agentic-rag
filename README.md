\# 🤖 Agentic RAG System



\[!\[Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

\[!\[Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

\[!\[Kubernetes](https://img.shields.io/badge/kubernetes-deployed-blue.svg)](https://kubernetes.io/)

\[!\[Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)



\## 📋 Overview



An \*\*intelligent Agentic RAG (Retrieval-Augmented Generation)\*\* system that uses an AI agent to decide when and how to retrieve information. Unlike traditional RAG systems that always search, this agent intelligently chooses between multiple tools based on the user's question.



\### 🎯 Key Features



\- \*\*🤖 Intelligent Agent\*\* - Decides WHEN to search vs answer directly

\- \*\*🔧 Multi-Tool Architecture\*\* - Policy search, web search, and calculator

\- \*\*🚀 Production Ready\*\* - Containerized with Docker, orchestrated with Kubernetes

\- \*\*📡 REST API\*\* - Easy integration with any application

\- \*\*⚡ Fast LLM\*\* - Powered by Groq's lightning-fast inference



\## 🏗️ Architecture



User Query → Agent Decision → Tool Selection → Response

↓

┌───────────┼───────────┐

↓ ↓ ↓

Policy Web Calculator

Search Search





\## 🚀 Quick Start



\### Local Development



```bash

\# Clone repository

git clone https://github.com/Aikaksh-Singh-Routela/agentic-rag.git

cd agentic-rag



\# Create virtual environment

python -m venv venv

source venv/bin/activate  # Linux/Mac

\# or

.\\venv\\Scripts\\activate  # Windows



\# Install dependencies

pip install -r requirements.txt



\# Set API key

export OPENAI\_API\_KEY="your-groq-api-key"



\# Run the API

python api\_server.py

