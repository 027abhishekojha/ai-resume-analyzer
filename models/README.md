# Models Directory

This directory stores trained ML model artifacts and fitted vectorizers.

## Structure

```
models/
├── trained/
│   └── classifier.joblib      ← Trained sklearn classifier
└── vectorizers/
    └── tfidf_vectorizer.joblib ← Fitted TF-IDF vectorizer
```

## Important

> ⚠️ **Model files are NOT tracked by Git** (see `.gitignore`).
> Train the model first: `python scripts/train_model.py`

## Model Versioning

For production use, consider:
- **MLflow**: Track experiments and model versions
- **DVC**: Version model artifacts alongside data
- **Model Registry**: Store versioned models in S3/GCS with metadata
