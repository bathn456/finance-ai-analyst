import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# 1. Page Config
st.set_page_config(page_title="AI Financial Analyst", layout="wide")
st.title("📊 AI Financial Analyst (10-K RAG)")

# 2. Sidebar for Keys
with st.sidebar:
    st.header("🔑 Credentials")
    st.info("Enter your keys to access the AI.")
    google_api_key = st.text_input("Google API Key", type="password")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password")
    st.markdown("---")
    st.markdown("Built by [Your Name]")

# 3. Main Logic
if google_api_key and pinecone_api_key:
    os.environ['GOOGLE_API_KEY'] = google_api_key
    os.environ['PINECONE_API_KEY'] = pinecone_api_key
    
    try:
        # Initialize Embeddings (Must be same model as ingestion!)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # Connect to Pinecone
        vectorstore = PineconeVectorStore(index_name="ragoffinance", embedding=embeddings)
        
        # Initialize LLM (Gemini 1.5 Flash - Free & Fast)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
        
        # Create Retrieval Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        # 4. Chat Interface
        st.write("### Ask a question about the Apple 2023 10-K")
        query = st.text_input("Example: What are the primary risk factors regarding China?")
        
        if query:
            with st.spinner("Analyzing document..."):
                response = qa_chain.invoke({"query": query})
                
                # Answer
                st.success("Analysis Complete")
                st.markdown(f"**Answer:** {response['result']}")
                
                # Evidence
                with st.expander("See Source Documents"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.write(f"**Source {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**")
                        st.info(doc.page_content)

    except Exception as e:
        st.error(f"Error connecting to AI: {e}")

else:
    st.warning("Please enter your API keys in the sidebar to start.")
