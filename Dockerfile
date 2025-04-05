# Dockerfile for Smart Semantic Search Engine

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose ports (8501 for Streamlit, 8000 for FastAPI)
EXPOSE 8501
EXPOSE 8000

# Default command: run FastAPI (can be overridden)
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
