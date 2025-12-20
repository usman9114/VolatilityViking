# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# gcc/g++ needed for some python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# 1. Install CPU-only PyTorch explicitly (Huge size savings: ~700MB vs 5GB+)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install other reqs (removing torch from reqs if present to avoid re-installing cuda version)
# We use grep to strip 'torch' from requirements.txt dynamically
RUN grep -v "torch" requirements.txt > requirements.no_torch.txt && \
    pip install --no-cache-dir -r requirements.no_torch.txt

# 3. Install extra dependencies (including dashboard)
RUN pip install --no-cache-dir sentence-transformers requests streamlit plotly

# Copy the rest of the application
COPY src/ src/
COPY scripts/ scripts/
COPY dashboard.py .
COPY dashboard_requirements.txt .
COPY data/raw/ data/raw/
COPY main.py .

# Create directories for data/logs/models
RUN mkdir -p data/processed data/logs models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8501

# Default command (can be overridden)
CMD ["python", "main.py", "--mode", "live", "--symbol", "ETH/USDT"]
