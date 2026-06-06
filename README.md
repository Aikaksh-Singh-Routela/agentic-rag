# 🤖 Agentic RAG System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-deployed-blue.svg)](https://kubernetes.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

An **intelligent Agentic RAG (Retrieval-Augmented Generation)** system that uses an AI agent to decide when and how to retrieve information. Unlike traditional RAG systems that always search, this agent intelligently chooses between multiple tools based on the user's question.

🔗 Links
GitHub: agentic-rag

Docker Hub: aikaksh/agentic-rag

### Key Features

| Feature | Description |
|---------|-------------|
| **🤖 Intelligent Agent** | Decides WHEN to search vs answer directly |
| **🔧 Multi-Tool Architecture** | Policy search, web search, and calculator |
| **🚀 Production Ready** | Containerized with Docker, orchestrated with Kubernetes |
| **📡 REST API** | Easy integration with any application |
| **⚡ Fast LLM** | Powered by Groq's lightning-fast inference |

## 🏗️ Architecture
User Query
↓
Agent Decision
↓
┌───────┼───────┐
↓ ↓ ↓
Policy Web Calculator
Search Search
↓
Response

text

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq Llama 3.1 8B |
| **Vector DB** | ChromaDB |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **API** | Flask |
| **Container** | Docker |
| **Orchestration** | Kubernetes |

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ask` | Ask a question |
| `GET` | `/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do I get?"}'
Example Response
json
{
  "answer": "You get 20 paid days off per year",
  "tool_used": "policy"
}
📦 Installation
Local Development
bash
# Clone repository
git clone https://github.com/Aikaksh-Singh-Routela/agentic-rag.git
cd agentic-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-groq-api-key"

# Run the API
python api_server.py
Docker
bash
# Pull from Docker Hub
docker pull aikaksh/agentic-rag:latest

# Run container
docker run -p 8080:8080 -e OPENAI_API_KEY="your-key" aikaksh/agentic-rag:latest
Kubernetes
bash
# Create secret
kubectl create secret generic groq-api-secret --from-literal=api-key="your-key"

# Deploy
kubectl apply -f k8s-deployment-local.yaml

# Get NodePort
kubectl get svc
📊 Sample Queries
Question	Tool Used	Response
"How many vacation days?"	Policy	20 paid days off per year
"Do I need MFA?"	Policy	MFA required for all system access
"What's 25 * 4?"	Direct	100
"Hello!"	Direct	Natural greeting

📄 License
MIT License

Built with 🤖, 🐳, and ☸️ by Aikaksh Singh Routela
