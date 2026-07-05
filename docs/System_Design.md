# 🔧 System Design — AI Resume Analyzer

## System Overview

The AI Resume Analyzer is a single-tier web application deployed as a Streamlit app. In its current form, all processing occurs on the server side within the Streamlit process.

---

## Component Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      Streamlit Server Process                   │
│                                                                  │
│  ┌──────────────┐     ┌───────────────────┐     ┌───────────┐  │
│  │  Streamlit   │────▶│  ResumeAnalyzer   │────▶│  Models   │  │
│  │  Web UI      │     │  (Orchestrator)   │     │  (.joblib)│  │
│  └──────────────┘     └────────┬──────────┘     └───────────┘  │
│                                │                                 │
│          ┌─────────────────────┼─────────────────────┐          │
│          ▼                     ▼                     ▼          │
│  ┌──────────────┐   ┌─────────────────┐   ┌────────────────┐   │
│  │ Preprocessing│   │Feature Enginrg  │   │   Evaluation   │   │
│  │ Pipeline     │   │(TF-IDF, KW ext) │   │   Visualizer   │   │
│  └──────────────┘   └─────────────────┘   └────────────────┘   │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## Scalability Considerations

| Concern | Current Approach | Future Approach |
|---|---|---|
| **Concurrency** | Streamlit session-based | Async FastAPI + worker pool |
| **Model Loading** | `st.cache_resource` (per-process) | Redis / model server (Triton) |
| **PDF Processing** | In-process, synchronous | Background task queue (Celery) |
| **Storage** | Local filesystem | S3 / GCS object storage |
| **Deployment** | Single process | Docker + Kubernetes |

---

## API Design (Future)

A REST API layer (FastAPI) is planned for enterprise use:

```
POST /api/v1/analyze
    Body: { resume_file: <PDF bytes>, job_description: <string> }
    Response: { similarity_score, prediction, keyword_gaps, ... }

GET /api/v1/health
    Response: { status: "healthy", model_version: "0.1.0" }

GET /api/v1/metrics
    Response: { model_accuracy, total_analyses, uptime_seconds }
```

---

## Security Considerations

- PDF files are processed in-memory (bytes), never written to disk
- No user data is persisted between sessions
- `.env` files contain secrets and are excluded from version control
- Model artifacts are excluded from Git (use DVC or cloud storage)
- Future API layer should implement rate limiting and authentication
