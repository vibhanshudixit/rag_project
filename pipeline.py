import os
from typing import List, Dict, Any
import streamlit as st
import torch

# PDF processing
from PyPDF2 import PdfReader

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

# Embeddings + LLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Retrieval + Reranking
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from dotenv import load_dotenv

load_dotenv('.env')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY') #type: ignore


device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                       model_kwargs={'device': device})



def load_documents(uploaded_files):
    """
    Convert uploaded files (Streamlit's UploadedFile objects or file paths) 
    into LangChain Document objects. Supports PDF and TXT.
    """
    all_docs = []
    os.makedirs('uploaded_docs', exist_ok=True)
    for uploaded_file in uploaded_files:
        #if isinstance(file, str):  # it's a file path
        file_path = os.path.join('uploaded_docs', uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        else:
            docs = []
        all_docs.extend(docs)
    return all_docs



def extract_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """Extract metadata like title, author, subject, producer, etc. from PDF."""
    reader = PdfReader(pdf_path)
    metadata = reader.metadata
    metadata_dict = {
        "title": getattr(metadata, "/Title", None),
        "author": getattr(metadata, "/Author", None),
        "subject": getattr(metadata, "/Subject", None),
        "creator": getattr(metadata, "/Creator", None),
        "producer": getattr(metadata, "/Producer", None),
        "keywords": getattr(metadata, "/Keywords", None),
        "num_pages": len(reader.pages),
        "file_path": pdf_path,
    }
    return metadata_dict


def load_and_chunk_pdf(pdf_path: str) -> List[Document]:
    documents = load_documents([pdf_path])
    semantic_splitter = SemanticChunker(
        embeddings,  
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )
    semantic_chunks = semantic_splitter.split_documents(documents)

    # Step 2: Recursive fallback for oversized chunks
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_chunks = []
    for doc in semantic_chunks:
        if len(doc.page_content) > 1200:  # if too large, break further
            sub_chunks = recursive_splitter.split_documents([doc])
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(doc)

    # Attach metadata (page number + file + pdf metadata)
    pdf_meta = extract_pdf_metadata(pdf_path)
    for chunk in final_chunks:
        chunk.metadata.update(pdf_meta)
        chunk.metadata["source"] = pdf_path
        chunk.metadata["page_number"] = chunk.metadata.get("page", 0)

    return final_chunks


def build_vectorstore(
    documents: List[Document],
    index_name: str = "company-index",
    collection_name: str = "rag_docs",
    **kwargs
):
    """Creates a Pinecone vectorstore from documents."""

    return PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name=index_name
    )


def get_retriever(docs, vectorstore, top_k: int = 5):
    """
    Creates a hybrid retriever with Pinecone (dense) + BM25 (sparse),
    then reranks results using Cohere Rerank.
    
    Args:
        docs: List of Documents (needed for BM25 retriever).
        vectorstore: Pinecone vectorstore instance.
        top_k: number of documents to retrieve.
    """

    # Sparse retriever (BM25 needs docs in-memory)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k

    # Dense retriever (Pinecone)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    # Hybrid retriever (ensemble of sparse + dense)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]   # adjust weights if needed
    )

    # Cohere Reranker
    reranker = CohereRerank(model="rerank-english-v3.0", top_n=top_k)

    # Contextual retriever (hybrid + rerank)
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever
    )

    return retriever

@st.cache_resource
def get_or_build_vectorstore(documents):
    """Cache the vectorstore so it doesn't rebuild on every query"""
    return build_vectorstore(documents)


class RAGPipeline:
    def __init__(self, documents, vectorstore, top_k = 10, index_name="rag-index"):
        self.index_name = index_name
        self.vectorstore = build_vectorstore(documents)
        self.retriever = get_retriever(vectorstore=self.vectorstore, docs=documents, top_k=top_k)
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    
    def query(self, question: str) -> Dict[str, Any]:
    # Retrieve top relevant docs
        docs = self.retriever.get_relevant_documents(question)

        # Collect sources with content + metadata
        sources = [
            {
                "content": d.page_content,
                "metadata": d.metadata
            }
            for d in docs
        ]

        # Build context with citation markers
        context_texts = []
        sources = []
        for i, doc in enumerate(docs, 1):
            snippet = doc.metadata.get("snippet_pointer", "")
            page = doc.metadata.get("page", "Unknown")
            context_texts.append(f"[{i}] {doc.page_content}")
            sources.append(f"[{i}] (Page {page}) Snippet: \"{snippet}\"")

        context_str = "\n\n".join(context_texts)
        
        prompt = f"""
    You are a helpful assistant. Use the provided sources to answer the question.
    Always include inline citations in square brackets, like [1], [2], etc., referring to the sources below.

    Question: {question}

    Sources:
    {context_str}

    Answer:
    """

        # Generate answer with inline citations
        answer = self.llm.predict(prompt)

        return {
            "answer": answer,
            "sources": sources
        }

