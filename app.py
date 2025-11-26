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

# --- IMPORT NEWS AGENT ---
try:
    from news_agent import get_company_news, analyze_news_sentiment
except ImportError:
    st.error("CRITICAL ERROR: `news_agent.py` is missing. Please create it in your GitHub repository.")
    st.stop()

# --- HELPER: SEARCH WITH DEBUGGING ---
def search_company(query, fmp_api_key, google_api_key):
    """
    Searches for a ticker. If it fails, it PRINTS the error to the screen.
    """
    clean_query = query.strip().upper()

    # 1. Direct Ticker Check (e.g. TSLA)
    url_profile = f"https://financialmodelingprep.com/api/v3/profile/{clean_query}?apikey={fmp_api_key}"
    try:
        response = requests.get(url_profile)
        
        # --- DEBUGGING START ---
        if response.status_code == 403:
            st.error("❌ FMP API Error 403: Forbidden. Your API Key might be invalid or expired.")
            return None, None
        if response.status_code == 429:
            st.error("❌ FMP API Error 429: Daily Limit Reached. Try again tomorrow or use a new key.")
            return None, None
            
        data = response.json()
        
        # Check for API error messages inside JSON
        if isinstance(data, dict) and "Error Message" in data:
            st.error(f"❌ FMP API Message: {data['Error Message']}")
            return None, None
        # --- DEBUGGING END ---

        if isinstance(data, list) and len(data) > 0:
            return data[0]['symbol'], data[0]['companyName']
            
    except Exception as e:
        st.error(f"Network connection error: {e}")

    # 2. AI Lookup (Fallback)
    if google_api_key:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=google_api_key)
            prompt = f"What is the stock ticker for '{query}'? Respond ONLY with the ticker symbol (e.g. AAPL)."
            ai_response = llm.invoke(prompt)
            predicted_ticker = ai_response.content.strip().upper().replace("\n", "").replace(" ", "")
            
            # Verify prediction with FMP
            url_verify = f"https://financialmodelingprep.com/api/v3/profile/{predicted_ticker}?apikey={fmp_api_key}"
            response = requests.get(url_verify)
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0]['symbol'], data[0]['companyName']
        except:
            pass

    return None, None

# --- HELPER: GET DATA ---
def get_company_data(ticker, api_key):
    metrics_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={api_key}"
    profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
    try:
        metrics = requests.get(metrics_url).json()
        profile = requests.get(profile_url).json()
        metrics_data = metrics[0] if isinstance(metrics, list) and metrics else {}
        profile_data = profile[0] if isinstance(profile, list) and profile else {}
        return {**metrics_data, **profile_data}
    except:
        return None

# --- HELPER: LEGAL DATA ---
def get_legal_data(company_name, api_key):
    if not api_key: return None
    url = f"https://api.opencorporates.com/v0.4/companies/search?q={company_name}&api_token={api_key}"
    try:
        response = requests.get(url).json()
        if response.get('results', {}).get('companies'):
            company = response['results']['companies'][0]['company']
            return {
                "name": company.get('name'),
                "jurisdiction": company.get('jurisdiction_code'),
                "incorporation_date": company.get('incorporation_date'),
                "source_url": company.get('opencorporates_url')
            }
    except:
        pass
    return None

# --- HELPER: PDF PROCESSING ---
def process_pdf(pdf_docs, api_key):
    raw_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# --- MAIN APP ---
def main():
    st.set_page_config(page_title="AI Financial Analyst", layout="wide")
    st.title("📊 AI Financial Analyst Dashboard")

    with st.sidebar:
        st.header("🔑 API Keys")
        google_api_key = st.text_input("Google API Key", type="password")
        fmp_api_key = st.text_input("FMP API Key", type="password")
        oc_api_key = st.text_input("OpenCorporates Key (Optional)", type="password")
    
    if not (google_api_key and fmp_api_key):
        st.warning("Enter Google and FMP keys to start.")
        return

    col1, col2 = st.columns([3, 1])
    search_query = col1.text_input("Enter Company Name:")
    
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None

    if col2.button("Analyze") and search_query:
        with st.spinner("Searching..."):
            # Pass both keys to the search function
            ticker, name = search_company(search_query, fmp_api_key, google_api_key)
            
            if ticker:
                st.session_state.current_ticker = ticker
                st.session_state.current_name = name
                st.session_state.company_data = get_company_data(ticker, fmp_api_key)
                st.session_state.legal_data = get_legal_data(name, oc_api_key)
            else:
                st.error(f"Could not find '{search_query}'. See error above.")

    if st.session_state.current_ticker:
        data = st.session_state.company_data
        name = st.session_state.current_name
        ticker = st.session_state.current_ticker
        
        st.markdown(f"## {name} ({ticker})")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Price", f"${data.get('price', 0)}")
        pe = data.get('peRatioTTM', 0)
        m2.metric("P/E Ratio", f"{pe if pe else 0:.2f}")
        m3.metric("Market Cap", f"${data.get('mktCap', 0):,}")

        if st.button("Generate AI Analysis"):
            with st.spinner("Analyzing..."):
                news = get_company_news(ticker)
                news_text = "\n".join(news) if news else "No news."
                
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
                prompt = f"Analyze {name} ({ticker}). P/E is {pe}. News: {news_text}. Is it undervalued? Answer in 3 bullets."
                res = llm.invoke(prompt)
                st.write(res.content)

        st.markdown("---")
        st.subheader("📄 Chat with 10-K")
        pdf = st.file_uploader("Upload Report", type="pdf")
        if pdf and st.button("Process PDF"):
            with st.spinner("Processing..."):
                st.session_state.vs = process_pdf([pdf], google_api_key)
                st.success("Ready!")

        if 'vs' in st.session_state:
            q = st.text_input("Ask a question:")
            if q:
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.vs.as_retriever())
                st.write(qa.invoke({"query": q})["result"])

if __name__ == '__main__':
    main()
