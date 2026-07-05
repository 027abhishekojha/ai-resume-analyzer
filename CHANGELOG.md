# 📝 Changelog

All notable changes to **AI Resume Analyzer** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- REST API layer using FastAPI
- Docker containerization and docker-compose setup
- Cloud deployment configuration (AWS / GCP / Azure)
- BERT/sentence-transformer integration for semantic similarity
- Named Entity Recognition using spaCy
- LLM-based resume feedback (GPT-4 / Gemini)
- Batch resume processing for enterprise use
- Multi-language resume support

---

## [0.1.0] - 2024-07-05

### Added
- Initial repository scaffold with production-ready structure
- PDF text extraction module using pdfplumber with PyPDF2 fallback
- NLP preprocessing pipeline (tokenization, stopword removal, lemmatization)
- TF-IDF feature engineering module
- Cosine similarity scoring engine
- ML inference pipeline (placeholder classifier)
- Streamlit UI — home page and results dashboard
- Keyword gap analysis module (placeholder)
- Configuration system via YAML + python-dotenv
- Logging configuration with rotating file handlers
- GitHub Actions CI pipeline (Python 3.12)
- Comprehensive documentation (README, Architecture, System Design, etc.)
- Pytest test suite for preprocessing, inference, and utilities
- Helper scripts: train_model.py, evaluate_model.py, download_dataset.py
- .gitignore, .env.example, pyproject.toml, requirements.txt

### Infrastructure
- Set up `src/` package structure with logical module separation
- Configured Black + Flake8 for code quality
- Added pre-commit hooks configuration in pyproject.toml

---

## [0.0.1] - 2024-07-01

### Added
- Initial project ideation and architecture planning
- Repository creation

---

[Unreleased]: https://github.com/yourusername/ai-resume-analyzer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/ai-resume-analyzer/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/yourusername/ai-resume-analyzer/releases/tag/v0.0.1
