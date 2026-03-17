# SEC / Annual Report RAG Q&A System

A Retrieval-Augmented Generation (RAG) system that lets you upload any company's annual report PDF and ask questions in plain English — with grounded answers and exact page citations.

## Live Demo

Upload a PDF → Ask questions → Get answers with page citations

---

## What it does

- Upload any annual report PDF (tested with Schneider Electric, TCS, Apple)
- Ask natural language questions: *"What was the revenue in 2023?"*
- Get grounded answers with exact page citations
- Follow-up questions work via conversation memory
- Ambiguous follow-ups are automatically rewritten for better retrieval

---

## Architecture

```
PDF Upload → pdfplumber → RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
          → SentenceTransformers (all-MiniLM-L6-v2) → ChromaDB

User Question → Query Rewriting → Embedding → ChromaDB Retrieval (top-5)
             → Groq LLaMA 3.1 8B → Grounded Answer + Page Citations
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq (LLaMA 3.1 8B Instant) |
| Orchestration | LangChain |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| PDF Parsing | pdfplumber |
| UI | Streamlit |
| Containerisation | Docker |

---

## Key Features

- **Semantic chunking** — splits at paragraph boundaries, not character count, preserving meaning across chunks
- **Caching** — same PDF is never re-ingested; MD5 hash check makes repeat queries instant
- **Query rewriting** — follow-up questions like *"how does that compare?"* are rewritten as standalone queries before retrieval
- **Citation system** — every answer is traceable to source page numbers shown below each response
- **Hallucination reduction** — LLM is instructed via system prompt to only use retrieved context, never generate figures from memory

---

## Evaluation

Custom LLM-as-judge evaluation framework (no external eval libraries) measuring:

| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer grounded in retrieved context? |
| Answer Relevancy | Does the answer address the question asked? |
| Correctness | Does the answer match ground truth? |
| Precision@K | Of K retrieved chunks, how many were actually relevant? |
| Recall@K | Of all relevant chunks, how many did retrieval find? |

---

## Setup

### Prerequisites

- Python 3.11+
- Groq API key — free at [console.groq.com](https://console.groq.com)

### Run locally

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/sec-rag-qa.git
cd sec-rag-qa
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
streamlit run app.py
```

### Run with Docker

```bash
docker build -t sec-rag .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key_here sec-rag
```

---

## Project Structure

```
sec-rag/
├── app/
│   ├── __init__.py
│   ├── ingest.py       # PDF parsing, chunking, ChromaDB ingestion
│   ├── retriever.py    # Semantic search over ChromaDB
│   ├── chain.py        # LangChain + Groq Q&A chain with memory + query rewriting
│   └── evaluate.py     # Custom RAG evaluation framework
├── app.py              # Streamlit UI
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Sample Questions to Try

- *What was the revenue in 2023?*
- *How did revenue change compared to last year?*
- *What are the key business risks?*
- *What is the company strategy going forward?*
- *What dividend was proposed?*

---

## Built by

Om — Research Scientist at IIT Bombay  
[LinkedIn](https://linkedin.com/in/YOUR_LINKEDIN) · [GitHub](https://github.com/YOUR_GITHUB_USERNAME)
