# app_streamlit.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import backend as b

st.set_page_config(page_title="Smart Semantic Search", page_icon="🔍", layout="wide")

# --- Header ---
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #1B5E20;'>🔍 Smart Semantic Search Engine</h1>
        <p style='font-size:18px; margin-top: -10px;'>Search Wikipedia, ArXiv, and Amazon with the power of NLP and Deep Learning!</p>
    </div>
    <hr style='border-top: 2px solid #4CAF50;'>
""", unsafe_allow_html=True)

# --- Inputs ---
st.markdown("""
    <div style='margin-top: 2rem;'>
""", unsafe_allow_html=True)
query = st.text_input("🔎 Enter your search query:", placeholder="e.g. Best headphones for work under 3000")
k = st.slider("📊 Number of results to display:", 1, 10, 5)
st.markdown("</div>", unsafe_allow_html=True)

# --- Real-Time Search ---
if query.strip():
    with st.spinner("⚡ Crunching data from multiple sources..."):
        wiki_titles = [
            "Artificial neural network",
            "Transformer (machine learning)",
            "Artificial intelligence",
            "Backpropagation"
        ]
        wiki_docs = b.scrape_wikipedia_pages(wiki_titles)
        arxiv_docs = b.scrape_arxiv(query, max_results=10)
        amazon_docs = b.load_amazon_data("amazon.csv", n=20)
        all_docs = wiki_docs + arxiv_docs + amazon_docs

        embeddings = b.embed_documents(all_docs)
        index = b.build_faiss_index(embeddings)
        results = b.semantic_hybrid_search(query, all_docs, index, b.model, k)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
        <h3 style='margin-bottom: 1rem; font-size: 22px;'>📄 Showing Top {k} Semantic Results for: <span style='color:#1B5E20;'>"{query}"</span></h3>
    """, unsafe_allow_html=True)

    for i, res in enumerate(results, start=1):
        st.markdown(f"""
            <div style='padding: 1.5rem; margin-bottom: 1.8rem; border-left: 6px solid #66BB6A; background-color: #f0fdf4; box-shadow: 2px 2px 6px rgba(0,0,0,0.05); border-radius: 8px;'>
                <h4 style='margin-bottom: 0.5rem; color: #2E7D32;'>{i}. {res['title']} <span style='color: grey; font-size: 0.85rem;'>({res['source']})</span></h4>
                <p style='font-size: 0.98rem; line-height: 1.65; text-align: justify;'>{res['text']}</p>
            </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <hr style='border-top: 1px solid #ccc;'>
    <div style='text-align: center; font-size: 14px; color: grey;'>
        🚀 Built with ❤️ using FastAPI, FAISS, SentenceTransformers, and Streamlit
    </div>
""", unsafe_allow_html=True)
