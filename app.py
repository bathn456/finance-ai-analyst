import streamlit as st
import os
import io
import requests
from pypdf import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- IMPORT NEWS AGENT (Safe Import) ---
try:
    from news_agent import get_company_news, analyze_news_sentiment
except ImportError:
    st.error("CRITICAL ERROR: `news_agent.py` is missing or cannot be imported. Please create it in your GitHub repository.")
    st.stop()

# --- HELPER: ROBUST COMPANY SEARCH ---
def search_company(query, api_key):
    """
    Searches for a ticker symbol by company name using FMP API.
    Includes a fallback to check if the query is already a valid ticker.
    """
    # Method 1: Search by Name (e.g., "Tesla")
    url = f"https://financialmodelingprep.com/api/v3/search?query={query}&limit=5&apikey={api_key}"
    try:
        response = requests.get(url).json()
        if response:
            return response[0]['symbol'], response[0]['name']
    except:
        pass

    # Method 2: Fallback - Direct Ticker Lookup (e.g., "TSLA")
    try:
        url_ticker = f"https://financialmodelingprep.com/api/v3/profile/{query.upper()}?apikey={api_key}"
        response = requests.get(url_ticker).json()
        if response:
             return response[0]['symbol'], response[0]['companyName']
    except:
        pass
        
    return None, None

# --- HELPER: GET FINANCIAL DATA ---
def get_company_data(ticker, api_key):
    """Fetches key metrics and profile data from FMP."""
    metrics_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={api_key}"
    profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
    
    try:
        metrics = requests.get(metrics_url).json()
        profile = requests.get(profile_url).json()
        
        # Handle cases where API returns empty lists
        metrics_data = metrics[0] if metrics else {}
        profile_data = profile[0] if profile else {}
        
        # Merge dictionaries
        return {**metrics_data, **profile_data}
    except Exception as e:
        return None

# --- HELPER: GET LEGAL DATA ---
def get_legal_data(company_name, api_key):
    """Fetches legal registration data from OpenCorporates."""
    if api_key:
        url = f"https://api.opencorporates.com/v0.4/companies/search?q={company_name}&api_token={api_key}"
    else:
        url = f"https://api.opencorporates.com/v0.4/companies/search?q={company_name}"
    
    try:
        response = requests.get(url).json()
        if response.get('results', {}).get('companies'):
            company = response['results']['companies'][0]['company']
            return {
                "name": company.get('name'),
                "number": company.get('company_number'),
                "jurisdiction": company.get('jurisdiction_code'),
                "address": company.get('registered_address_in_full'),
                "incorporation_date": company.get('incorporation_date'),
                "status": company.get('current_status'),
                "source_url": company.get('opencorporates_url')
            }
        return None
    except:
        return None

# --- HELPER: PDF PROCESSING ---
def process_pdf(pdf_docs, api_key):
    """Reads, chunks, and embeds uploaded PDFs into a FAISS vector store."""
    raw_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=api_key
    )
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# --- MAIN APP UI ---
def main():
    st.set_page_config(page_title="Batuhan's Financial Deck", layout="wide")
    st.title("📊 AI Financial Analyst Dashboard")

    # Sidebar for Keys
    with st.sidebar:
        st.header("🔑 API Keys")
        google_api_key = st.text_input("Google API Key", type="password")
        fmp_api_key = st.text_input("FMP API Key", type="password")
        oc_api_key = st.text_input("OpenCorporates Key (Optional)", type="password")
        st.markdown("---")
        st.info("Enter a company name (e.g. 'Tesla') or ticker (e.g. 'TSLA') to begin.")

    # Check for required keys
    if not (google_api_key and fmp_api_key):
        st.warning("Please enter Google and FMP API keys to start.")
        return

    # 1. SEARCH SECTION
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Enter Company Name:")
    with col2:
        search_btn = st.button("Analyze Company")

    # Initialize Session State
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None
        st.session_state.current_name = None
        st.session_state.company_data = None
        st.session_state.legal_data = None

    # Logic: Execute Search
    if search_btn and search_query:
        with st.spinner(f"Searching markets for '{search_query}'..."):
            # Search for Ticker
            ticker, name = search_company(search_query, fmp_api_key)
            
            if ticker:
                st.session_state.current_ticker = ticker
                st.session_state.current_name = name
                # Fetch Financials
                st.session_state.company_data = get_company_data(ticker, fmp_api_key)
                # Fetch Legal Data
                st.session_state.legal_data = get_legal_data(name, oc_api_key)
            else:
                st.error("Company not found. Try using the exact ticker symbol (e.g. AAPL).")

    # 2. DISPLAY DASHBOARD
    if st.session_state.current_ticker and st.session_state.company_data:
        data = st.session_state.company_data
        legal = st.session_state.legal_data
        name = st.session_state.current_name
        ticker = st.session_state.current_ticker

        # Header
        st.markdown(f"## 🏢 {name} ({ticker})")
        st.markdown(f"*{data.get('description', 'No description available')[:300]}...*")
        
        # --- METRICS ---
        col_metrics, col_legal = st.columns([2, 1])
        
        with col_metrics:
            st.subheader("📈 Financial Snapshot")
            m1, m2, m3 = st.columns(3)
            m1.metric("Price", f"${data.get('price', 0)}")
            
            pe = data.get('peRatioTTM', 0)
            # Handle None/Zero PE
            if pe is None: pe = 0
            
            m2.metric("P/E Ratio", f"{pe:.2f}")
            m3.metric("Market Cap", f"${data.get('mktCap', 0):,}")
            
            # Simple Valuation Logic
            if pe > 0:
                if pe < 15:
                    st.success("✅ Potentially Undervalued (Low P/E)")
                elif pe > 30:
                    st.error("⚠️ Potentially Overvalued (High P/E)")
                else:
                    st.info("⚖️ Fairly Valued (Average P/E)")
            else:
                st.write("P/E Not Applicable (Negative Earnings)")

        with col_legal:
            st.subheader("⚖️ Legal Identity")
            if legal:
                st.write(f"**Jurisdiction:** {legal['jurisdiction'].upper() if legal.get('jurisdiction') else 'N/A'}")
                st.write(f"**Incorporated:** {legal['incorporation_date']}")
                st.write(f"**Status:** {legal['status']}")
                if legal.get('source_url'):
                    st.markdown(f"[View Official Record]({legal['source_url']})")
            else:
                st.warning("No legal record found.")

        st.markdown("---")

        # --- AI COMPREHENSIVE VALUATION ---
        st.subheader("🧠 AI Strategic Valuation")
        if st.button("Generate Comprehensive Analysis"):
            with st.spinner("Gathering intelligence (Financials + News)..."):
                # 1. Fetch News using the agent
                news_list = get_company_news(ticker)
                news_text = "\n".join(news_list) if news_list else "No recent news found."
                
                # 2. Build the Master Prompt
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    temperature=0.7, 
                    google_api_key=google_api_key
                )
                
                analysis_prompt = f"""
                You are a senior hedge fund manager. Perform a comprehensive valuation analysis for **{name} ({ticker})**.
                
                **Data Source 1: Quantitative Financials**
                - P/E Ratio: {pe} (Sector Average is approx 20-25)
                - Market Cap: ${data.get('mktCap', 0):,}
                - Current Price: ${data.get('price', 0)}
                - Industry: {data.get('industry', 'Unknown')}
                
                **Data Source 2: Qualitative Market Sentiment (Recent News)**
                {news_text}
                
                **Task:**
                1. Synthesize the financial metrics with the news sentiment.
                2. Determine if the company appears **Undervalued**, **Overvalued**, or **Fairly Valued**.
                3. Explain your reasoning clearly. Does the news justify the current P/E ratio?
                """
                
                res = llm.invoke(analysis_prompt)
                st.write(res.content)

        st.markdown("---")

        # --- PDF CHAT ---
        st.subheader(f"📄 Chat with {ticker}'s Annual Report")
        st.info("Optional: Upload a 10-K or Annual Report to ask specific questions.")
        
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True, key="pdf_uploader")
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.session_state.vectorstore = process_pdf(pdf_docs, google_api_key)
                    st.success("Knowledge Base Ready!")
            else:
                st.error("Upload a file first.")

        if 'vectorstore' in st.session_state:
            user_question = st.text_input(f"Ask a question about {name}:")
            if user_question:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    temperature=0, 
                    google_api_key=google_api_key
                )
                
                # Create prompt with legal context if available
                legal_context = ""
                if legal:
                    legal_context = f"Legal Context: Incorporated {legal.get('incorporation_date')} in {legal.get('jurisdiction')}."
                
                template = f"""
                You are a financial analyst analyzing **{name} ({ticker})**.
                {legal_context}
                Use the context below to answer.
                
                Context: {{context}}
                Question: {{question}}
                """
                
                PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
                
                qa = RetrievalQA.from_chain_type(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=st.session_state.vectorstore.as_retriever(), 
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                res = qa.invoke({"query": user_question})
                st.write(res["result"])

if __name__ == '__main__':
    main()
