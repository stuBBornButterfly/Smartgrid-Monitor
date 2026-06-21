# SmartGrid Monitor — runs FastAPI (port 8000, internal) + Streamlit (port 7860, public)
FROM python:3.11-slim

# HF Spaces run as a non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /home/user/app

# Install CPU-only torch first (small), then the rest
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --user --no-cache-dir -r requirements.txt

# Copy code + model artifacts
COPY --chown=user . .

EXPOSE 7860

# Start the API in the background, wait a moment, then launch the dashboard
CMD ["bash", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & sleep 8 && streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0"]
