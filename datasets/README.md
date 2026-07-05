# 📂 Datasets

This directory stores training and evaluation datasets for the AI Resume Analyzer.

## Directory Structure

```
datasets/
├── raw/           ← Original, unmodified downloaded data
└── processed/     ← Cleaned, schema-validated data ready for training
```

## Important Notes

> ⚠️ **Dataset files are NOT tracked by Git** (see `.gitignore`).
> Large data files should be managed via DVC or stored in cloud storage (S3/GCS).

## Getting Started

Generate a synthetic dataset for development:

```bash
python scripts/download_dataset.py --samples 500
```

## Dataset Schema (processed/)

| Column | Type | Description |
|---|---|---|
| `resume_text` | str | Pre-processed resume text |
| `jd_text` | str | Pre-processed job description text |
| `label` | int (0/1) | 1 = Suitable, 0 = Not Suitable |

## Recommended Public Datasets

- [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- [Job Description Dataset](https://www.kaggle.com/datasets/andrewmvd/data-scientist-jobs)
