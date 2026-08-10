<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.110+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Flask-2.3+-lightgrey.svg" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

# 🧪 API-Lab

**A collection of production‑grade and experimental APIs for LLM applications, document processing, and agent workflows.**

This repository is a practical toolkit for developers and data scientists building AI‑powered services. It contains:

- 🔹 **Basic Flask endpoints** to get started with chatbot, RAG, and agent patterns.
- 🔹 **Advanced FastAPI services** that are complex, industry‑ready, and solve real‑world problems — including an LLM Gateway, Intelligent Document Processing, Semantic Cache, Content Moderation, and more.

Whether you're prototyping or deploying to production, you'll find a solid foundation here.

---

## 📖 Table of Contents

- [Features](#-features)
- [API Inventory](#-api-inventory)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Environment Variables](#-environment-variables)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

- **Multi‑framework** – Both Flask and FastAPI examples, so you can choose your style.
- **Production ready** – Many FastAPI services include authentication, rate limiting, caching, background workers, and webhooks.
- **Modular & extensible** – Each API is self‑contained; plug in your own models, vectors, or LLM providers.
- **Comprehensive** – Covers chatbots, RAG, agents, semantic caching, LLM gateways, document processing, content moderation, and more.
- **Well documented** – Every file includes detailed docstrings and usage examples.

---

## 📦 API Inventory

### 🟢 Flask Examples (Legacy & Learning)

| File | Description |
|------|-------------|
| `basic_chatbot_api.py` | Minimal `/api/chat` that forwards a query to `llmchain.chain`. |
| `simple_LangGraph_chatbot_api.py` | Uses a LangGraph `workflow.graph` to process queries. |
| `simple_RAG_chatbot_api.py` | RAG example: retrieves from `vectorstore.db`, reranks with `CrossEncoder`, calls `llmchain.chain`. |
| `Human_In_the_Loop_api.py` | LangGraph workflow with human‑interrupt endpoints (`/api/chat`, `/api/human-response`). |
| `Supervise_MultiAI_Agent_API.py` | Multi‑agent supervisor that invokes `multi_agent.graph` with structured results. |

### 🚀 FastAPI Services (Production‑Grade)

| File | Description |
|------|-------------|
| `LLM_Gateway_api.py` | 🛡️ **LLM Gateway** – Unified interface for OpenAI, Anthropic, Gemini with cost control, PII redaction, rate limiting, and semantic caching. |
| `Intelligent_Document_Processing_api.py` | 📄 **IDP** – Upload PDFs/images; OCR, classify (invoice/receipt/contract), extract structured fields, validate, and notify via webhook. |
| `Semantic_Cache_api.py` | 💾 **Semantic Cache** – Caches LLM responses using embedding similarity to reduce cost and latency. |
| `Content_Moderation_api.py` | 🚫 **Content Moderation** – Scans text for toxicity, PII, and policy violations using both rule‑based and LLM‑based checks. |
| `Structured_extraction_api.py` | 🧩 **Structured Extraction** – Extracts entities, relationships, and JSON from unstructured text (e.g., emails, support tickets). |
| `Customer_Support_RAG.py` | 💬 **Customer Support RAG** – End‑to‑end RAG pipeline with retrieval, reranking, and answer generation, tailored for support queries. |
| `ToolCalling_Agent_api.py` | 🛠️ **Tool‑Calling Agent** – An agent that can call external tools (APIs, calculators, etc.) with memory and state. |
| `Document_Ingestion_Pinecone_api.py` | 📂 **Document Ingestion** – Ingests documents, chunks them, and indexes into Pinecone (or any vector DB) for scalable retrieval. |
| `Retrieval_eval_api.py` | 📊 **Retrieval Evaluation** – Evaluates retrieval quality using metrics like hit rate, MRR, and NDCG. |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- `pip` and `virtualenv` (recommended)
- For OCR (IDP): Tesseract‑OCR installed ([instructions](#optional-ocr-dependency))

### 1️⃣ Clone the repository

```bash
git clone https://github.com/raj-tembe/API-Lab.git
cd API-Lab
```

### 2️⃣ Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

Install common dependencies (adjust as needed for specific APIs):

```bash
pip install fastapi "uvicorn[standard]" flask flask-cors python-dotenv pydantic-settings \
    openai anthropic google-generativeai sentence-transformers langchain-google-genai \
    langchain-huggingface pytesseract pdfplumber pillow aiofiles httpx python-multipart
```

> **Note:** Some APIs require additional libraries. Check the docstring of each file for exact requirements.

### 4️⃣ Set environment variables

Create a `.env` file in the project root. At a minimum, set your API keys:

```env
# For LLM Gateway
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# For administration (if required)
ADMIN_API_KEY=your_admin_secret

# For Semantic Cache
GOOGLE_API_KEY=your_gemini_key
```

Refer to each API’s docstring for all available settings.

### 5️⃣ Run an API

Each API is self‑contained; just run the file:

```bash
# Flask example
python basic_chatbot_api.py

# FastAPI example
python LLM_Gateway_api.py

# Or using uvicorn directly
uvicorn LLM_Gateway_api:app --reload --port 8000
```

Most FastAPI services expose interactive docs at `http://localhost:8000/docs`.

---

## 📝 Usage Examples

### Basic Flask Chatbot

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, how are you?"}'
```

### LLM Gateway (FastAPI)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test_team_123" \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"team":"engineering"}'
```

### Intelligent Document Processing

```bash
curl -X POST http://localhost:8000/api/v1/documents \
  -H "X-API-Key: test_team_123" \
  -F "file=@invoice.pdf" \
  -F "webhook_url=https://webhook.site/abc123"
```

### Semantic Cache

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your return policy?"}'
```

---

## ⚙️ Environment Variables

Many APIs support configuration via `.env` variables. Common ones:

| Variable | Description | Example |
|----------|-------------|---------|
| `ADMIN_API_KEY` | Admin key for management endpoints | `admin_secret` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `GEMINI_API_KEY` | Gemini API key | `...` |
| `GOOGLE_API_KEY` | Used by Semantic Cache | `...` |
| `DEFAULT_DAILY_BUDGET` | Daily budget for teams (LLM Gateway) | `100.0` |
| `RATE_LIMIT_PER_MINUTE` | Requests per minute per team | `60` |
| `UPLOAD_DIR` | Directory for uploaded documents (IDP) | `./uploads` |

Check each API’s `Settings` class for a full list.

---

## 🌐 Optional OCR Dependency

For `Intelligent_Document_Processing_api.py`, OCR requires Tesseract:

- **Ubuntu**: `sudo apt install tesseract-ocr`
- **Mac**: `brew install tesseract`
- **Windows**: Download installer from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

If OCR is not installed, the API will fall back to a mock text extractor.

---

## 🤝 Contributing

Contributions are welcome! If you have a new API idea, a bug fix, or an improvement:

1. Fork the repository.
2. Create a feature branch.
3. Add your code with clear docstrings and examples.
4. Open a pull request.

Please ensure your code follows the existing style (PEP 8) and includes appropriate documentation.

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

**Happy building!** 🎉 If you find this useful, consider starring the repository ⭐.
