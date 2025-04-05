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

def load_amazon_data(path="amazon.csv", n=20):
    df = pd.read_csv(path)
    df["description"] = df[["About Product", "Technical Details", "Product Specification"]].fillna("").agg(" ".join, axis=1)
    df = df.dropna(subset=["Product Name"])
    df = df[df["description"].str.strip() != ""]
    docs = []
    for i, row in df.head(n).iterrows():
        docs.append({
            "id": f"amazon_{i}",
            "title": row["Product Name"],
            "source": "amazon",
            "text": row["description"]
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
def load_data_and_index():
    wiki_titles = [
        "Artificial neural network",
        "Transformer (machine learning)",
        "Artificial intelligence",
        "Backpropagation"
    ]
    wiki_docs = scrape_wikipedia_pages(wiki_titles)
    arxiv_docs = scrape_arxiv("transformer", max_results=10)
    amazon_docs = load_amazon_data("amazon.csv", n=20)
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

# Preload data once
all_docs, index = load_data_and_index()

@app.get("/search", response_model=List[SearchResult])
def search(query: str = Query(..., description="Your semantic query"), k: int = 5):
    return semantic_hybrid_search(query, all_docs, index, model, k)

# --- Run server ---
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
