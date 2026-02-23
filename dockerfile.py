FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit>=1.35.0

# Copy app
COPY . .

# HF Space expects port 7860
EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
