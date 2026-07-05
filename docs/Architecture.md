# 🏗️ Architecture — AI Resume Analyzer

## Overview

The AI Resume Analyzer follows a **layered clean architecture** to separate concerns and maintain testability, scalability, and maintainability.

---

## Architectural Layers

```
┌──────────────────────────────────────────┐
│         Presentation Layer               │  src/app/
│   Streamlit UI, input handling, charts  │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│         Business / Orchestration Layer   │  src/inference/analyzer.py
│   Coordinates pipeline, manages state   │
└─────────────────┬────────────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐  ┌─────────────────────┐
│  ML Layer     │  │  Preprocessing Layer │
│  inference/   │  │  preprocessing/      │
│  training/    │  │  feature_engineering/│
│  evaluation/  │  └─────────────────────┘
└───────────────┘
        │
┌───────▼──────────────────────────────────┐
│         Utilities Layer                  │  src/utils/
│  Config loader, Logger, File helpers    │
└──────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────┐
│         Configuration Layer              │  configs/
│  config.yaml, logging.yaml, .env        │
└──────────────────────────────────────────┘
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `src/app/` | Streamlit UI, user input, result rendering |
| `src/preprocessing/` | PDF extraction, text cleaning, tokenization |
| `src/feature_engineering/` | TF-IDF vectorization, keyword extraction |
| `src/inference/` | Cosine similarity, classification, full pipeline |
| `src/training/` | Dataset loading, model training, cross-validation |
| `src/evaluation/` | Metrics computation, visualization |
| `src/utils/` | Config loading, logging, file I/O, text helpers |

---

## Data Flow

```
User Uploads PDF + Pastes JD
           │
           ▼
    [PDF Extractor]          pdfplumber → PyPDF2 fallback
           │
           ▼
    [Text Cleaner]           Remove URLs, special chars, normalize
           │
           ▼
    [Tokenizer]              word_tokenize → stopword removal → lemmatize
           │
     ┌─────┴─────┐
     ▼           ▼
[Resume Vec] [JD Vec]        TF-IDF transform (fitted vectorizer)
     │           │
     └─────┬─────┘
           ▼
  [Cosine Similarity]        sklearn.metrics.pairwise.cosine_similarity
           │
           ├─── [Score Label]  Excellent / Good / Fair / Poor
           │
  [ML Classifier]            LogisticRegression.predict_proba()
           │
           ├─── [Prediction]  Suitable / Not Suitable
           │
  [Keyword Gap Analyzer]     JD top-k keywords not in resume
           │
           ▼
    [Results Dashboard]      Streamlit + Plotly charts
```

---

## Key Design Decisions

1. **Lazy Loading of ML Artifacts**: The vectorizer and model are loaded once at app startup using `st.cache_resource`, avoiding repeated I/O.

2. **Fallback Extraction**: pdfplumber is the primary PDF extractor due to its layout-awareness. PyPDF2 serves as a fallback for edge cases.

3. **Configurable Thresholds**: All scoring thresholds (similarity bands, classification threshold) are defined in `configs/config.yaml`, not hardcoded.

4. **Separation of Inference and Training**: The `inference/` and `training/` modules are completely independent. The app only imports from `inference/`.

5. **Stateless Pipeline**: Each analysis call is stateless — no session state persists between calls except cached ML artifacts.
