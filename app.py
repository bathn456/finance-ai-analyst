import streamlit as st
import os
import io
# Import for reading PDF bytes (pypdf is installed as a dependency)
from pypdf import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Google Embeddings and LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# FAISS for in-memory, session-based vector storage
from langchain_community.vectorstores import FAISS 
from langchain.chains import RetrievalQA

# --- FUNCTION: DOCUMENT PROCESSING (Ingestion) ---
def get_vectorstore_from_pdf(pdf_docs, api_key):
    """Loads PDFs, extracts text, chunks it, creates embeddings, and stores in FAISS."""
    raw_text = ""
    for pdf in pdf_docs:
        # Use io.BytesIO to read the uploaded file from memory
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
            
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    text_chunks = text_splitter.split_text(raw_text)

    # Create embeddings and FAISS index (Memory)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", # 768 Dimensions
        google_api_key=api_key
    )
    
    # Store in FAISS in-memory index
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# --- MAIN STREAMLIT APPLICATION ---
def main():
    # 1. Page Configuration and Title
    st.set_page_config(page_title="Batuhan Yilmaz's AI Analyst", layout="wide")
    st.title("🤖 Batuhan Yilmaz's AI Document Analyst")

    # 2. Sidebar for Credentials and Upload
    with st.sidebar:
        st.header("🔑 Credentials & Documents")
        
        # Google API Key input
        google_api_key = st.text_input("Google API Key", type="password")
        
        # File Uploader
        st.subheader("Upload Document")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files (e.g., 10-K, Report)", 
            accept_multiple_files=True, 
            type=['pdf']
        )
        process_button = st.button("Submit & Process")
        st.markdown("---")
        st.info("Built by Batuhan Yilmaz with Google Gemini, LangChain, & Streamlit.")

    # Check for API Key
    if not google_api_key:
        st.warning("Please enter your Google API key in the sidebar to begin processing documents.")
        return

    # 3. Processing Logic (Runs only on button click)
    if process_button:
        if pdf_docs:
            with st.spinner("Processing PDF and generating vector index..."):
                try:
                    # Clear any previous index and create the new one
                    st.session_state.vectorstore = get_vectorstore_from_pdf(pdf_docs, google_api_key)
                    st.session_state.processing_done = True
                    st.success("Documents Processed! Ask a question below.")
                except Exception as e:
                    st.error(f"Error during processing or embedding: {e}")
                    st.session_state.processing_done = False
        else:
            st.error("Please upload at least one PDF file before pressing Submit.")

    # 4. Chat UI (Appears only after processing is complete)
    if 'vectorstore' in st.session_state and st.session_state.get('processing_done', False):
        st.write("### Ask a Question about the Uploaded Material")
        user_question = st.text_input("Enter your financial query here:")

        if user_question:
            with st.spinner("Generating accurate, context-aware answer..."):
                # LLM setup (Fast, free-tier model)
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    temperature=0,
                    google_api_key=google_api_key
                )

                # RetrievalQA uses the FAISS index stored in session_state
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(),
                    return_source_documents=True # Get the source text for verification
                )
                
                response = qa_chain.invoke({"query": user_question})
                
                # Display Answer
                st.markdown(f"**Answer:** {response['result']}")
                
                # Display Sources
                with st.expander("📄 See Source Documents (Evidence)"):
                    for i, doc in enumerate(response["source_documents"]):
                        # FAISS doesn't track pages from uploaded PDFs easily, so we just show the content
                        st.info(f"Source Snippet {i+1}:\n{doc.page_content[:500]}...")

if __name__ == '__main__':
    main()
