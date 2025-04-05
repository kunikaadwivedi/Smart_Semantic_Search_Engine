# app_streamlit.py (Beautiful FAANG-style Design 🌈)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import backend as b

st.set_page_config(page_title="Smart Semantic Search", page_icon="🔍", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e8f5e9, #f1f8e9);
    }
    .result-card:hover {
        box-shadow: 0px 0px 12px rgba(76, 175, 80, 0.4);
        transform: scale(1.01);
        transition: all 0.2s ease-in-out;
    }
    a {
        text-decoration: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #1B5E20;'>🔍 Smart Semantic Search Engine</h1>
        <p style='font-size:18px; margin-top: -10px;'>
            Explore knowledge from <b>Wikipedia 🌐</b>, <b>ArXiv 📚</b>, and <b>Amazon 🛒</b> using semantic search powered by AI.
        </p>
    </div>
    <hr style='border-top: 2px solid #4CAF50;'>
""", unsafe_allow_html=True)

# --- Inputs ---
st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
query = st.text_input("🔎 Enter your search query:", placeholder="e.g. Best headphones for work under 3000")
k = st.slider("📊 Number of results to display:", 1, 10, 5)
st.markdown("</div>", unsafe_allow_html=True)

# --- Real-Time Search ---
if query.strip():
    with st.spinner("⚡ Crunching data from multiple sources..."):
        amazon_docs = b.fetch_amazon_products(query, n=10)
        results = b.semantic_hybrid_search(query, amazon_docs, b.static_docs, b.static_embeddings, b.model, k=k)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
        <h3 style='margin-bottom: 1rem; font-size: 22px;'>📄 Top {k} Semantic Results for: <span style='color:#1B5E20;'>"{query}"</span></h3>
    """, unsafe_allow_html=True)

    icon_map = {"amazon": "🛒", "arxiv": "📚", "wikipedia": "🌐"}

    for i, res in enumerate(results, start=1):
        icon = icon_map.get(res['source'].lower(), "🔎")
        st.markdown(f"""
            <div class='result-card' style='padding: 1.5rem; margin-bottom: 1.8rem; border-left: 6px solid #66BB6A; background-color: #f0fdf4; box-shadow: 2px 2px 8px rgba(0,0,0,0.04); border-radius: 10px;'>
                <h4 style='margin-bottom: 0.5rem; color: #2E7D32;'>
                    {i}. <a href="{res.get('url', '#')}" target="_blank">{res['title']}</a>
                    <span style='color: grey; font-size: 0.85rem;'>({icon} {res['source'].capitalize()})</span>
                </h4>
                <p style='font-size: 0.97rem; line-height: 1.6; text-align: justify;'>{res['text']}</p>
            </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <hr style='border-top: 1px solid #ccc;'>
    <div style='text-align: center; font-size: 14px; color: grey;'>
        🚀 Built with ❤️ by Kunikaa Dwivedi using FastAPI, FAISS, SentenceTransformers, and Streamlit · v1.0
        <br>
        <a href='https://www.linkedin.com/in/kunikaa-dwivedi' target='_blank'>🔗 Connect on LinkedIn</a>
    </div>
""", unsafe_allow_html=True)
