FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py
RUN mkdir -p /app/cache /app/offline /app/logs

# Cloud platforms expect 8080; expose it for clarity
EXPOSE 8080

# Bind to all interfaces and use $PORT (Cloud Run sets this)
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
