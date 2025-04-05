# app.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import wikipedia
import urllib.parse
import feedparser
import faiss
import warnings
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import uvicorn
import requests

warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

# Load SBERT model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Scrapers ---
def scrape_wikipedia_pages(titles):
    docs = []
    for title in titles:
        try:
            summary = wikipedia.summary(title, sentences=5)
            docs.append({
                "id": f"wiki_{title.replace(' ', '_')}",
                "title": title,
                "source": "wikipedia",
                "text": summary
            })
        except Exception as e:
            print(f"Failed to fetch '{title}': {e}")
            continue
    return docs

def scrape_arxiv(search_query="machine learning", max_results=10):
    query = urllib.parse.quote(search_query)
    base_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    feed = feedparser.parse(base_url)
    docs = []
    for entry in feed.entries:
        docs.append({
            "id": f"arxiv_{entry.id.split('/')[-1]}",
            "title": entry.title,
            "source": "arxiv",
            "text": entry.summary
        })
    return docs

def fetch_amazon_products(query, n=10):
    url = "https://api.rainforestapi.com/request"
    params = {
        "api_key": "22FE831FD329439D837C1A0EA8358A96",  # Replace this with your Rainforest API key
        "type": "search",
        "amazon_domain": "amazon.in",
        "search_term": query
    }
    response = requests.get(url, params=params).json()
    docs = []
    for i, product in enumerate(response.get("search_results", [])[:n]):
        docs.append({
            "id": f"amazon_{i}",
            "title": product.get("title", ""),
            "source": "amazon",
            "text": product.get("snippet", product.get("title", ""))
        })
    return docs

# --- Embedding + Index ---
def embed_documents(docs):
    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# --- Load Everything ---
def load_data_and_index(query):
    wiki_titles = [
        "Artificial neural network",
        "Transformer (machine learning)",
        "Artificial intelligence",
        "Backpropagation"
    ]
    wiki_docs = scrape_wikipedia_pages(wiki_titles)
    arxiv_docs = scrape_arxiv(query, max_results=10)
    amazon_docs = fetch_amazon_products(query, n=10)
    all_docs = wiki_docs + arxiv_docs + amazon_docs

    embeddings = embed_documents(all_docs)
    index = build_faiss_index(embeddings)
    return all_docs, index

# --- Semantic Search ---
def semantic_hybrid_search(query, docs, index, model, k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    results = []
    for i in indices[0]:
        results.append({
            "title": docs[i]["title"],
            "source": docs[i]["source"],
            "text": docs[i]["text"][:300]
        })
    return results

# --- FastAPI App ---
app = FastAPI(title="Smart Semantic Search Engine")

class SearchResult(BaseModel):
    title: str
    source: str
    text: str

@app.get("/search", response_model=List[SearchResult])
def search(query: str = Query(..., description="Your semantic query"), k: int = 5):
    all_docs, index = load_data_and_index(query)
    return semantic_hybrid_search(query, all_docs, index, model, k)

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
