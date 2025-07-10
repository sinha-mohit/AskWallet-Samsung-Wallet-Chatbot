# Base image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
COPY .env .
COPY ask_wallet.py .
COPY vectorstore/ vectorstore/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "ask_wallet.py", "--server.port=8501", "--server.address=0.0.0.0"]
