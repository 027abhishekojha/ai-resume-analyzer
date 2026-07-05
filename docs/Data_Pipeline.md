# 📊 Data Pipeline — AI Resume Analyzer

## Overview

The data pipeline transforms raw PDF resumes and job description text into structured numerical features ready for ML model inference.

---

## Pipeline Stages

### Stage 1: Input Ingestion
- **Source**: User-uploaded PDF (Streamlit file uploader) + pasted JD text
- **Validation**: File type, size, non-empty checks before processing

### Stage 2: PDF Text Extraction
- **Primary**: `pdfplumber` — handles multi-column layouts, tables
- **Fallback**: `PyPDF2` — simpler but wider format support
- **Output**: Raw text string (may contain noise)

### Stage 3: Text Cleaning
- Lowercase conversion
- URL and email removal (regex)
- Special character stripping
- Unicode normalization (NFKD → ASCII)
- Whitespace normalization

### Stage 4: Tokenization
- `nltk.word_tokenize` for sentence-aware tokenization
- Stopword removal (NLTK English corpus)
- WordNet lemmatization (reduce inflected forms)
- Minimum token length filter

### Stage 5: TF-IDF Vectorization
- **Vectorizer**: `sklearn.TfidfVectorizer`
- **Vocabulary**: Fitted on training corpus (5,000 features max)
- **N-grams**: Unigrams + bigrams (1, 2)
- **Scaling**: Sublinear TF (log normalization)
- **Output**: Dense numpy arrays of shape `(1, 5000)`

### Stage 6: Similarity & Classification
- **Cosine Similarity**: `sklearn.metrics.pairwise.cosine_similarity`
- **Classification**: `LogisticRegression.predict_proba`

### Stage 7: Keyword Analysis
- Extract top-K keywords from JD by TF-IDF weight
- Cross-reference against resume token set
- Gaps = JD keywords not found in resume

---

## Training Data Schema

```csv
resume_text,jd_text,label
"experienced python developer...", "looking for python engineer...", 1
"marketing manager creative...", "data scientist machine learning...", 0
```

| Column | Type | Description |
|---|---|---|
| `resume_text` | str | Pre-processed resume text |
| `jd_text` | str | Pre-processed job description text |
| `label` | int (0/1) | 1 = Suitable, 0 = Not Suitable |

---

## Data Quality Checks

- [ ] Duplicate detection (text hash comparison)
- [ ] Minimum word count per document (>50 words)
- [ ] Label distribution validation (flag severe class imbalance)
- [ ] Language detection (English-only filter)

---

## Recommended Dataset Sources

| Source | Description | License |
|---|---|---|
| [Kaggle Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) | 2400+ labeled resumes | Open |
| [Resume-Job Match Dataset](https://huggingface.co/datasets) | HuggingFace Hub datasets | Varies |
| Custom scraping | LinkedIn / Indeed (check ToS) | Proprietary |
