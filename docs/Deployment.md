# 🚀 Deployment Guide — AI Resume Analyzer

## Local Development

```bash
# 1. Clone and set up
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic dataset and train the model
python scripts/download_dataset.py
python scripts/train_model.py

# 3. Launch the app
streamlit run src/app/main.py
```

---

## Docker Deployment (Planned)

```dockerfile
# Dockerfile (placeholder — not yet implemented)
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "src/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t ai-resume-analyzer .
docker run -p 8501:8501 ai-resume-analyzer
```

---

## Cloud Deployment Options

### Streamlit Community Cloud (Easiest)
1. Push to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository and set `src/app/main.py` as the entry point
4. Add secrets via the Streamlit Secrets manager

### AWS EC2
```bash
# On EC2 instance (Ubuntu 22.04)
sudo apt update && sudo apt install python3.12 python3-pip -y
git clone <your-repo-url>
cd ai-resume-analyzer
pip install -r requirements.txt
streamlit run src/app/main.py --server.port 80
```

### Google Cloud Run (Serverless)
```bash
gcloud run deploy ai-resume-analyzer \
  --image gcr.io/PROJECT_ID/ai-resume-analyzer \
  --platform managed \
  --port 8501 \
  --region us-central1
```

---

## Environment Variables in Production

| Variable | Description | Required |
|---|---|---|
| `MODEL_PATH` | Path to trained classifier | Yes |
| `VECTORIZER_PATH` | Path to fitted vectorizer | Yes |
| `LOG_LEVEL` | Logging verbosity | No (default: INFO) |

---

## Health Checks

Current: None (Streamlit handles this internally).

Future (FastAPI): `GET /api/v1/health` endpoint.

---

## Performance Benchmarks (Placeholder)

| Metric | Value |
|---|---|
| PDF parsing time | ~0.5–1.5s per page |
| Preprocessing time | ~0.1–0.3s |
| TF-IDF transform time | ~0.05s |
| End-to-end inference | <2s |
| Streamlit cold start | ~5–10s |
