import os
import tempfile

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Ensure the parent directory is in the Python path if running directly
import sys
from pathlib import Path

from rag import vector_store
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from utils.logger import get_logger
from rag.embeddings.embedding_manager import EmbeddingManager
from rag.vector_store.chroma_store import ChromaVectorStore
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.preprocessor import TextPreprocessor
from rag.ingestion.chunker import DocumentChunker
from rag.llm.llm_factory import LLMFactory
from rag.retrieval.retriever import RetrieverManager
from rag.llm.chain import RAGChain

logger = get_logger(__name__)

# --- Page Config ---
st.set_page_config(
    page_title="Intelligent RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    .source-box {
        font-size: 0.8rem;
        background-color: #1E2127;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 3px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            AIMessage(content="Hello! I'm your Intelligent RAG Assistant. Please upload some documents in the sidebar and enter your Groq API key to get started.")
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None


def process_uploaded_files(uploaded_files, api_key):
    """Process uploaded files and initialize the RAG pipeline."""
    if not uploaded_files:
        st.warning("Please upload at least one document.")
        return False
        
    if not api_key:
        st.warning("Please enter your Groq API key.")
        return False
        
    # Update settings with API key temporarily for this session
    os.environ["GROQ_API_KEY"] = api_key
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 1. Initialize Embeddings and Vector Store
        status_text.text("Initializing embedding model...")
        embedding_manager = EmbeddingManager()
        vector_store = ChromaVectorStore(embedding_manager.embeddings)
        progress_bar.progress(20)
        
        # 2. Process Files
        status_text.text("Loading and chunking documents...")
        all_chunks = []
        chunker = DocumentChunker()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
                
            try:
                # Load
                docs = DocumentLoader.load_file(temp_path)
                # Preprocess
                docs = TextPreprocessor.process_documents(docs)
                # Chunk
                chunks = chunker.split_documents(docs)
                
                # Update metadata for display
                for chunk in chunks:
                    chunk.metadata["source_name"] = uploaded_file.name
                    
                all_chunks.extend(chunks)
            finally:
                os.unlink(temp_path) # Clean up temp file
                
        progress_bar.progress(60)
        
        # 3. Add to Vector Store
        if all_chunks:
            status_text.text("Clearing previous documents...")
            vector_store.clear_collection()

            status_text.text(f"Adding {len(all_chunks)} chunks to vector store...")
            vector_store.clear_collection()
            vector_store.add_documents(all_chunks)

            st.session_state.vector_store = vector_store
        
        # 4. Initialize LLM and Chain
        status_text.text("Initializing LLM and RAG Chain...")
        llm = LLMFactory.create_llm(api_key=api_key)
        retriever_manager = RetrieverManager(vector_store)
        retriever = retriever_manager.get_retriever()
        
        rag_chain = RAGChain(llm, retriever)
        st.session_state.rag_chain = rag_chain
        
        progress_bar.progress(100)
        status_text.text("Setup complete! You can now ask questions.")
        return True
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return False


def main():
    settings = get_settings()
    init_session_state()
    
    # --- Sidebar ---
    with st.sidebar:
        st.title("🧠 Setup & Config")
        st.markdown("Configure your assistant and upload documents.")
        
        # API Key input
        api_key_input = st.text_input(
            "Groq API Key", 
            type="password", 
            value=settings.groq_api_key or "",
            help="Get your API key from https://console.groq.com/"
        )
        
        st.divider()
        
        # File Upload
        st.subheader("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD files", 
            type=["pdf", "txt", "md"], 
            accept_multiple_files=True
        )
        
        if st.button("Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                success = process_uploaded_files(uploaded_files, api_key_input)
                if success:
                    st.success("Documents processed successfully!")
                    
        st.divider()
        st.markdown("### Settings")
        st.caption(f"Model: {settings.default_model}")
        st.caption(f"Embeddings: {settings.embedding_model}")
        st.caption(f"Chunk Size: {settings.chunk_size}")

    # --- Main Chat Area ---
    st.title("Intelligent RAG Assistant")
    st.markdown("Ask questions about your uploaded documents.")
    
    # Display Chat History
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                # If there's sources attached to the AI message in metadata
                if hasattr(message, "additional_kwargs") and "sources" in message.additional_kwargs:
                    with st.expander("View Sources"):
                        for source in message.additional_kwargs["sources"]:
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source:</strong> {source.get('name', 'Unknown')}<br/>
                                <em>"{source.get('content', '')[:200]}..."</em>
                            </div>
                            """, unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        # Display user message
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Check if chain is initialized
        if st.session_state.rag_chain is None:
            with st.chat_message("assistant"):
                st.warning("Please upload documents and process them first before asking questions.")
                st.session_state.messages.append(AIMessage(content="Please upload documents and process them first before asking questions."))
            return
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    answer = response["answer"]
                    context_docs = response.get("context", [])
                    
                    st.markdown(answer)
                    
                    # Prepare sources
                    sources_data = []
                    if context_docs:
                        with st.expander("View Sources"):
                            for i, doc in enumerate(context_docs):
                                source_name = doc.metadata.get("source_name", doc.metadata.get("source", f"Document {i+1}"))
                                content_preview = doc.page_content.replace("\n", " ").strip()
                                
                                sources_data.append({
                                    "name": source_name,
                                    "content": content_preview
                                })
                                
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source:</strong> {source_name}<br/>
                                    <em>"{content_preview[:200]}..."</em>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Store AI response with sources in metadata
                    ai_message = AIMessage(
                        content=answer, 
                        additional_kwargs={"sources": sources_data} if sources_data else {}
                    )
                    st.session_state.messages.append(ai_message)
                    
                except Exception as e:
                    logger.error(f"Error generating answer: {e}", exc_info=True)
                    error_msg = f"Sorry, I encountered an error while trying to answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append(AIMessage(content=error_msg))

if __name__ == "__main__":
    main()
