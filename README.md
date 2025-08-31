### README.md
# Hybrid RAG App (Streamlit + Pinecone + Cohere ReRank)

This project is a **Retrieval-Augmented Generation (RAG)** app built with **Streamlit**.  
It lets you upload PDF or text documents, indexes them into Pinecone, and provides a chatbot-style interface that answers your questions using a **hybrid retriever** (dense + sparse) with **Cohere Reranking** and **Groq LLMs**.


To access the frontend on StreamLit Cloud: https://ragproject-vibhanshu.streamlit.app/


---

## Features
- Upload **PDF** or **TXT** documents directly in the Streamlit app.
- Extracts and chunks document text intelligently using **LangChain** (semantic + recursive splitting).
- Builds a **hybrid retriever**:
  - Dense retrieval with **Pinecone + HuggingFace embeddings**
  - Sparse retrieval with **BM25**
  - **Cohere Rerank** for final scoring
- Query documents with inline citations, answers, and metadata.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/hybrid-rag-app.git
cd hybrid-rag-app
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root with your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key
COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## Usage

Run the Streamlit app:

```bash
streamlit run frontend.py
```

Then open the link (usually `http://localhost:8501/`) in your browser.

---

## Project Structure

```
.
├── frontend.py        # Streamlit frontend
├── pipeline.py        # Document processing & RAG pipeline
├── requirements.txt   # Dependencies
├── README.md          # Project documentation
└── .env               # API keys (not committed)
```

---
## Project Architecture

```text
                    ┌─────────────────────────┐
                    │      User Uploads       │
                    │   (PDFs, Docs, etc.)    │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Document Processing     │
                    │  (Chunking & Embedding)  │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │     Pinecone VectorDB    │
                    │  (Stores embeddings for  │
                    │   semantic retrieval)    │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    Query Processing      │
                    │  (Retrieval + Re-ranking │
                    │   + Context Formatting)  │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    LLM (Groq / OpenAI)   │
                    │ Generates final response │
                    └─────────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Streamlit Frontend     │
                    │   (User-facing answers)  │
                    └─────────────────────────┘
'''text

## Notes

* GPU acceleration (CUDA) will be used automatically if available.
* Uploaded documents are saved locally under `uploaded_docs/`.
* Pinecone index name defaults to `rag-index` but can be adjusted in `pipeline.py`.

---

## Example Workflow

1. Start the app (`streamlit run frontend.py`)
2. Upload one or more PDFs/TXT files
3. Ask a question about the documents
4. Get an AI-generated answer with inline citations and retrieved metadata

---

## Tech Stack

* [Streamlit](https://streamlit.io/) – UI
* [LangChain](https://www.langchain.com/) – Document loading, splitting, retrieval
* [Pinecone](https://www.pinecone.io/) – Vector database
* [Cohere](https://cohere.com/) – Reranking
* [Groq](https://groq.com/) – LLM inference
* [HuggingFace Transformers](https://huggingface.co/) – Embeddings

---


Got it 👍 — here’s a **more concise, human-friendly** version (2–3 points each):

---

###  Remarks

* **Limits**:

  - Since we rely on Pinecone, there are API limits, costs, and occasional latency issues.
  - Chunking large documents sometimes loses important context.
  - Retrieval quality is only as good as the embeddings + reranker we use.

* **Trade-offs**:

  - Pinecone gives us scalability and persistence, but it’s more expensive than local options like FAISS.
  - Hybrid retrieval (Pinecone + BM25 + Cohere ReRank) improves accuracy, but adds extra complexity and latency.
  - Modular design makes it flexible to extend, but production setups would need tighter control and monitoring.

* **Next Steps**:

  - Tune Pinecone for a better balance of cost, speed, and accuracy.
  - Add caching so repeated queries don’t keep hitting Pinecone.
  - Explore smarter pipelines (like document-type awareness or GraphRAG) to make retrieval more context-aware.

---



