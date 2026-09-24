<img width="1191" height="341" alt="image" src="https://github.com/user-attachments/assets/8d1216b1-596c-4301-8e6e-106f4cbb1d6b" /># OmniContext Local Intelligence Framework

An air-gapped, enterprise-grade local vector Retrieval-Augmented Generation (RAG) framework optimized for edge execution on Snapdragon NPU/CPU hardware architectures. Built for the Snapdragon® AI Lab Build & Present Challenge 2026.

---

## Key Features
* **Air-Gapped & Secure:** 100% offline execution; zero cloud telemetry or data leakage, ensuring total enterprise/academic privacy.
* **Local Vector RAG Engine:** Chunks raw documents, computes mathematical embeddings using `nomic-embed-text`, and indexes them locally.
* **Semantic Vector Search:** Uses NumPy-accelerated cosine similarity to retrieve exact context paragraphs instantly.
* **SQLite Persistence:** Commits all document records, vectors, and query history to a local `omnictext_vector.db` audit store.
* **Dual-Mode Enterprise Workflows:** Seamlessly toggles between *Academic Research Mode* and *Workplace Financial Audit Mode* via dynamic system prompt routing.

---

## Technical Architecture
1. **Ingestion Layer:** Parses multi-page local PDFs or text files via `pypdf`.
2. **Embedding & Vector Store:** Generates semantic float arrays via local Ollama (`nomic-embed-text`) and stores chunks in SQLite.
3. **Retrieval Core:** Computes cosine similarity matching against user queries to isolate top-scoring context blocks.
4. **Inference Engine:** Executes local model inference through Ollama (`qwen2.5:3b`) targeted at local hardware accelerators.

---

## Prerequisites & Installation

### 1. Prerequisites
* Python (3.10+) installed
* Ollama running locally on your machine.

### 2. Pull Required Local Models
Run these commands in your terminal:
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```
### 3. Setup Repository & Install Dependencies
```bash
git clone [https://github.com/YOUR_USERNAME/omnicontext-framework.git](https://github.com/YOUR_USERNAME/omnicontext-framework.git)
cd omnicontext-framework
py -m pip install -r requirements.txt
```

### 4. Launch application
py -m streamlit run app.py









