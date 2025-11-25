import streamlit as st
import tempfile 
import os
import io # New import for handling file bytes
from pypdf import PdfReader # New import for reading PDF bytes
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI 
from langchain_community.vectorstores import FAISS # New: In-Memory Vector Store
from langchain.chains import RetrievalQA

# ... (API Key input unchanged) ...

# --- NEW INGESTION FUNCTION ---
def get_vectorstore_from_pdf(pdf_docs):
    raw_text = ""
    for pdf in pdf_docs:
        # Use io.BytesIO to read the uploaded file from memory
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
            
    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(raw_text)

    # Create embeddings and FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    # ... (Page config and API key logic) ...

    # 🔑 Credentials & LLM setup (unchanged)
    if not (google_api_key): # Pinecone key is not needed for FAISS
        st.warning("Please enter your Google API key to proceed.")
        return

    # --- FILE UPLOADER UI ---
    with st.sidebar:
        st.header("Upload Document")
        # Streamlit widget for file upload
        pdf_docs = st.file_uploader(
            "Upload your PDF Files", 
            accept_multiple_files=True, 
            type=['pdf']
        )
        process_button = st.button("Submit & Process")

    # --- PROCESSING LOGIC ---
    if process_button and pdf_docs:
        with st.spinner("Processing PDF and generating memory..."):
            # This calls the function above to create the memory
            st.session_state.vectorstore = get_vectorstore_from_pdf(pdf_docs)
            st.success("Documents Processed! Ask a question below.")
    
    # --- CHAT UI ---
    if 'vectorstore' in st.session_state:
        # LLM setup
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        user_question = st.text_input("Ask a Question about your uploaded file:")

        if user_question:
            # RetrievalQA uses the FAISS index stored in session_state
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.vectorstore.as_retriever(),
            )
            response = qa_chain.invoke({"query": user_question})
            st.write(response["result"])
    
if __name__ == '__main__':
    main()
