from cryptonews.crypto_news import get_crypto_news, all_ticker_news, get_trending_headlines, get_event_articles, get_top_mentioned_crypto_tickers
import streamlit as st

st.title("Welcome to Donkey Betz AI")
st.markdown("&nbsp;", unsafe_allow_html=True) 




# Adds caching to prevent extra api calls
@st.cache_resource

def display_crypto_news():
    news_list = get_crypto_news()
    for news in news_list:
        title = st.text(news['title'])
        cols = st.columns(2)
        cols[0].image(news['image_url'])
        cols[1].markdown(f"[{news['title']}]({news['news_url']})")
        cols[1].write(f"Date: {news['date']}")
        cols[1].write(f"Source: {news['source_name']}")
        cols[1].write(news['text'])
        cols[1].write(f"Sentiment: {news['sentiment']}")


display_crypto_news()

