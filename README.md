# 🚀 Gradio Hybrid RAG Assistant

An advanced, production-grade Retrieval-Augmented Generation (RAG) system built to query and synthesize Gradio documentation. This project features a custom ingestion pipeline, a hybrid local/cloud architecture, and a rigorous LLM-as-a-Judge evaluation framework to mathematically prove factual accuracy.

## 🧠 Architecture Overview

This assistant is built on a two-part architecture designed for accuracy, speed, and cost-efficiency:

1. **The Retrieval Pipeline (Local Vector DB)**
   * **Knowledge Base:** Raw Markdown documentation chunks.
   * **Enhancement:** Uses local LLMs (via Ollama) to dynamically generate headers and summaries for each chunk prior to embedding, dramatically improving semantic search recall.
   * **Vector Store:** ChromaDB for fast, local similarity search.

2. **The Generation Pipeline (Cloud Inference)**
   * **Query Rewriting:** Transforms raw user queries into optimized search vectors.
   * **Synthesis:** Uses fast, lightweight models to generate highly technical, accurate responses based strictly on retrieved context.
   * **UI/UX:** Built natively in Gradio using `gr.ChatInterface`, featuring custom HTML accordions that allow users to transparently inspect the exact retrieved source chunks used for the answer.

## 📊 Evaluation & Metrics (LLM-as-a-Judge)

To mathematically prove the system's reliability, I built a custom **LLM-as-a-Judge Evaluation Pipeline**. This bypasses heavy black-box frameworks (like Ragas) in favor of transparent, Python-native metric calculations.

To strictly avoid **Self-Preference Bias**, the evaluation pipeline uses a cross-model grading architecture. The lightweight generator's answers are evaluated by a frontier reasoning model (**GPT-4o**) against a golden dataset of 50 synthetic, highly technical test cases.

**Key Performance Indicators:**
* **Mean Reciprocal Rank (MRR):** `0.8184` *(Indicates the correct context is consistently retrieved in the top 1 or 2 slots).*
* **Normalized DCG (nDCG):** `0.7976` *(Indicates highly relevant chunks are successfully ranked at the top of the context window).*
* **Factual Accuracy:** `4.50 / 5.0`
* **Answer Completeness:** `4.64 / 5.0`
* **Answer Relevance:** `4.82 / 5.0`

> *Note: A custom Gradio analytics dashboard (`dashboard.py`) is included in this repository to interactively visualize these metrics and inspect the detailed logs.*

## ⚙️ Quick Start

### Prerequisites
* Python 3.10+
* An OpenAI API Key
* [Optional] Ollama installed locally for ingestion chunk summarization

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/Shailya777/gradio-rag-assistant.git](https://github.com/Shailya777/gradio-rag-assistant.git)
   cd gradio-rag-assistant
   ```
2. Install the required dependencies:
    ```bash
   pip install -r requirements.txt
   ```
3. Create a .env file in the root directory and add your API key:
    ```bash
    OPENAI_API_KEY=your_api_key_here
    ```

### Running the Application
1. Launch the Assistant Interface:
    ```bash
    python app.py
    ```
2. Launch the Evaluation Dashboard:
    ```bash
    python dashboard.py
    ```