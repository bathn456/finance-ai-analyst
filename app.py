import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

def get_company_news(ticker):
    """
    Fetches the latest news for a given ticker using Yahoo Finance.
    Returns a list of formatted news strings.
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return None
            
        formatted_news = []
        for item in news[:5]: # Limit to latest 5 articles to save tokens
            title = item.get('title', 'No Title')
            publisher = item.get('publisher', 'Unknown Source')
            link = item.get('link', '#')
            formatted_news.append(f"- {title} (Source: {publisher})")
            
        return formatted_news
    except Exception as e:
        print(f"Error fetching news: {e}")
        return None

def analyze_news_sentiment(ticker, news_list, api_key):
    """
    Uses Gemini to analyze the sentiment of the fetched news.
    """
    if not news_list:
        return "No recent news found to analyze."

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.7, 
        google_api_key=api_key
    )

    # Create a single string of news
    news_text = "\n".join(news_list)

    template = """
    You are a financial journalist and market sentiment analyst. 
    Here are the latest news headlines for **{ticker}**:

    {news_text}

    Please provide a concise analysis:
    1. What is the overall market sentiment? (Bullish/Bearish/Neutral)
    2. What are the key drivers mentioned?
    3. Highlight any immediate risks or opportunities.
    """

    prompt = PromptTemplate(
        input_variables=["ticker", "news_text"],
        template=template
    )

    chain = prompt | llm
    response = chain.invoke({"ticker": ticker, "news_text": news_text})
    
    return response.content
