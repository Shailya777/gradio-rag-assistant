# 🤖 Gradio RAG Expert Assistant

> An advanced Retrieval-Augmented Generation (RAG) assistant for Gradio documentation, featuring query rewriting, semantic chunking, and LLM-as-a-judge evaluation.

This project adapts advanced RAG architectures to build a highly accurate, domain-specific AI assistant. It is designed to navigate the official Gradio documentation, intelligently chunk Python code alongside explanatory text, and provide precise answers to complex UI development questions.

This system is configured to support both cloud-based LLMs (OpenAI) and local inference (via Ollama) to leverage local hardware acceleration for cost-free document ingestion and synthetic evaluation.

## 🗂️ Project Architecture
1. **Data Ingestion (`ingest.py`):** Semantically chunks Markdown documentation, generating LLM-powered headlines and summaries for each chunk to improve retrieval accuracy, stored in ChromaDB.
2. **Retrieval & Answer (`answer.py` & `app.py`):** Features query rewriting (to optimize semantic search) and re-ranking of context chunks before generating a final response via a Gradio UI.
3. **Synthetic Evaluation (`evaluator.py`):** Utilizes an LLM-as-a-judge approach to score retrieval metrics (MRR, nDCG) and answer quality (Accuracy, Completeness, Relevance) against a generated dataset (`tests.jsonl`).

---

## 🛠️ Phase 1: Environment Setup

This project uses Conda to manage dependencies, ensuring the smooth installation of vector database requirements (like ChromaDB) on Windows/WSL environments.

**1. Create the Conda Environment**
```bash
conda create -n gradio_rag_env python=3.11 -y
```

**2. Activate the Environment**
```bash
conda activate gradio_rag_env
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```
(Note for IDE Users: Ensure your IDE Python Interpreter is pointed to this existing Conda environment to enable proper code completion and linting.)


## Phase 2: Data Acquisition
To ensure the assistant is providing accurate, up-to-date information, the knowledge base is pulled directly from the official Gradio GitHub repository.

**1. Clone the Gradio Repository**
To keep the project cleanly separated, clone the Gradio repo into a directory outside of this project folder:
```bash
cd ..
git clone [https://github.com/gradio-app/gradio.git](https://github.com/gradio-app/gradio.git)
```

**2. Extract Core Documentation**
- Navigate to the cloned repository and open the guides folder (gradio/guides/). We only want the core English .md text files for the embedding model.

- Create a knowledge-base folder in the root of this project (gradio-rag-assistant/knowledge-base/).

- Copy the numbered guide folders (e.g., 01_getting-started, 02_building-interfaces, through 11_other_tutorials).

- Crucial Exclusion Rules:
❌ Do not copy the assets/ folder (Contains images and CSS, which the text embedding model cannot process).
❌ Do not copy the cn/ folder (Contains zh-CN localized translations, which will duplicate context and confuse the retriever).

Paste the selected numbered folders directly into your knowledge-base directory.


## Phase 3: Execution Sprint Plan

1. Day 1: Ingestion & Embeddings

- Adapt the ingest.py script to parse the extracted Gradio Markdown files.

- Configure the embedding model (text-embedding-3-large or local equivalent).

- Run the ingestion pipeline to populate the local preprocessed_db via ChromaDB.

2. Day 2: Pipeline & Interface

- Update answer.py system prompts to reflect the Gradio Assistant persona.

- Validate the query rewriting and re-ranking logic against the new ChromaDB collection.

- Launch the chat interface via app.py and conduct manual Q&A testing.

3. Day 3: Evaluation & Refinement

- Generate a tests.jsonl file containing 20-30 complex Gradio development questions, required keywords, and reference answers.

- Run evaluator.py to calculate MRR, nDCG, and Answer Quality metrics.

- Refine chunking size/overlap if retrieval metrics fall below the 80% threshold.
