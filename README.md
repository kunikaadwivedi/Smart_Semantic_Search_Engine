# 🔍 Smart Semantic Search Engine

A production-ready, FAANG-style semantic search engine that lets you query Wikipedia, ArXiv, and Amazon product data using natural language. Built with FastAPI, Streamlit, Sentence Transformers, and FAISS.

---

## 🚀 Features

- ✅ Real-time semantic search
- ✅ Integrated Wikipedia, ArXiv, and Amazon sources
- ✅ SBERT embeddings with FAISS indexing
- ✅ Streamlit frontend for instant UX
- ✅ Modular FastAPI backend
- ✅ Docker-ready & deployable to the cloud

---

## 📁 Project Structure

```
smart-semantic-search/
├── backend/
│   ├── api.py               # FastAPI app
│   └── pipeline.py          # Scraper, embedder, search engine
├── app_streamlit.py         # Streamlit frontend
├── app.py                   # Unified backend (optional)
├── requirements.txt         # Python dependencies
└── README.md                # You are here
```

---

## 📦 Installation

### ⚙️ Create Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 📥 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 🧪 Run the Backend
```bash
uvicorn backend.api:app --reload
```

### 💻 Run the Frontend
```bash
streamlit run app_streamlit.py
```

---

## 🐳 Run with Docker (optional)

Build the image:
```bash
docker build -t smart-search .
```

Run Streamlit app:
```bash
docker run -p 8501:8501 smart-search streamlit run app_streamlit.py
```

---

## 🤖 Technologies Used
- Python 3.10
- FastAPI
- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- pandas, wikipedia, feedparser

---

## 🧠 Credits
Built with ❤️ by Kunikaa Dwivedi

Let's connect on [LinkedIn](https://www.linkedin.com/in/kunikaa-dwivedi-429610242/) 🔗

---

## ⭐ Showcase This Project
- Add it to your GitHub with a short video demo
- Deploy on Render or Hugging Face Spaces
- Mention in your resume as:

> Built a scalable NLP-based semantic search engine using real-time embedding and vector similarity, integrated across multiple data sources.
