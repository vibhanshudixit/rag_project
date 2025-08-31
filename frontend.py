import streamlit as st

# Import your pipeline
from pipeline import RAGPipeline, load_documents, get_or_build_vectorstore


# File uploader
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    documents = load_documents(uploaded_files)   # Prepare docs here
    vectorstore = get_or_build_vectorstore(documents)   # Feed into your vectorstore
    st.session_state.rag = RAGPipeline(documents, vectorstore)
    st.success("Vectorstore ready!")
else: 
    st.info("Please Upload documents to start!")

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(index_name="rag-index", top_k=5, documents=documents, vectorstore=vectorstore)

st.set_page_config(page_title="Hybrid RAG App", layout="wide")

st.title("Hybrid RAG with Pinecone + Cohere ReRank")
st.markdown("Ask questions about your uploaded documents. You'll see the answer, citations, and metadata.")
query = st.text_input("Enter your question:")

if st.button("Run Query") and query:
    with st.spinner("Fetching answer..."):
        response = st.session_state.rag.query(query)

        # Display final answer
        st.subheader("Answer")
        st.write(response["answer"])
        if response.get("citations"):
            st.markdown("### Citations")
            for idx, citation in enumerate(response["citations"], 1):
                pdf_path = citation["pdf_path"]
                page_num = citation.get("page", "N/A")
                text_snippet = citation.get("snippet", "")

                st.markdown(f"**[{idx}]** Page {page_num} — {pdf_path}")
                st.write(text_snippet)
                st.write("---")

        # Retrieved document metadata
        if response.get("metadata"):
            st.markdown("### Retrieved Document Metadata")
            for i, meta in enumerate(response["metadata"], 1):
                st.json(meta)
