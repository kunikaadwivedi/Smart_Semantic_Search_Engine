# app.py (Optimized for speed ⚡)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import wikipedia
import urllib.parse
import feedparser
import faiss
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import uvicorn
import requests

warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

# Load fast SBERT model once
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# --- Static Scrapers for ArXiv + Wikipedia ---
def scrape_wikipedia_pages(titles):
    docs = []
    for title in titles:
        try:
            summary = wikipedia.summary(title, sentences=5)
            docs.append({
                "id": f"wiki_{title.replace(' ', '_')}",
                "title": title,
                "source": "wikipedia",
                "text": summary,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
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
            "text": entry.summary,
            "url": entry.link
        })
    return docs

# --- Realtime Amazon Fetch ---
def fetch_amazon_products(query, n=10):
    url = "https://api.rainforestapi.com/request"
    params = {
        "api_key": "22FE831FD329439D837C1A0EA8358A96",
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
            "text": product.get("snippet", product.get("title", "")),
            "url": product.get("link", "")
        })
    return docs

# --- Embedding + Indexing ---
def embed_documents(docs):
    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# --- Preload Static Docs (Wikipedia + ArXiv) ---
wiki_titles = [
    "Artificial neural network",
    "Transformer (machine learning)",
    "Artificial intelligence",
    "Backpropagation"
]
static_docs = scrape_wikipedia_pages(wiki_titles) + scrape_arxiv("transformer")
static_embeddings = embed_documents(static_docs)

# --- Semantic Search ---
def semantic_hybrid_search(query, amazon_docs, k=5):
    all_docs = static_docs + amazon_docs
    amazon_embeddings = embed_documents(amazon_docs)
    all_embeddings = np.vstack([static_embeddings, amazon_embeddings])
    index = build_faiss_index(all_embeddings)
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    results = []
    for i in indices[0]:
        results.append({
            "title": all_docs[i]["title"],
            "source": all_docs[i]["source"],
            "text": all_docs[i]["text"][:300],
            "url": all_docs[i].get("url", "#")
        })
    return results

# --- FastAPI App ---
app = FastAPI(title="Smart Semantic Search Engine")

class SearchResult(BaseModel):
    title: str
    source: str
    text: str
    url: str

@app.get("/search", response_model=List[SearchResult])
def search(query: str = Query(..., description="Your semantic query"), k: int = 5):
    amazon_docs = fetch_amazon_products(query, n=10)
    return semantic_hybrid_search(query, amazon_docs, k)

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
