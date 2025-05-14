FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential curl libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy files
COPY app.py ./app.py
COPY src/models ./src/models

# Install Python dependencies
RUN pip install --no-cache-dir streamlit xgboost pandas scikit-learn joblib

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]