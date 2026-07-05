# 💡 Learnings — AI Resume Analyzer

## Key Technical Learnings

### 1. Clean Architecture Pays Off
Separating preprocessing, feature engineering, inference, and the UI into distinct modules made the codebase dramatically easier to test and extend. When the tokenizer logic needed changes, only `tokenizer.py` and its tests needed updating — no changes cascaded into the UI layer.

### 2. Configuration Over Hardcoding
Centralizing all thresholds and hyperparameters in `configs/config.yaml` with environment variable overrides made it trivial to experiment with different settings without modifying source code. This is essential for ML projects where you tune constantly.

### 3. Fallback Strategies in I/O Pipelines
PDF parsing is inherently fragile. Building explicit primary/fallback patterns (pdfplumber → PyPDF2) with clear logging at each stage made debugging broken uploads far simpler.

### 4. Type Hints and Docstrings Are Not Optional
In a project with multiple interacting modules (preprocessing → feature engineering → inference), clear type signatures and docstrings serve as living documentation that IDEs and code reviewers rely on. They also make writing tests far easier.

### 5. ML Projects Need Data Early
The hardest lesson: starting with architecture first is fine for planning, but ML projects live or die by data quality. Acquiring or generating labeled training data should start in parallel with code architecture, not after it.

### 6. Streamlit's `cache_resource` Is Powerful
Using `st.cache_resource` to cache the vectorizer and model across sessions avoided re-loading 50+ MB artifacts on every user request, making the app feel responsive.

### 7. scikit-learn Pipelines Are Worth Learning
Future iterations will wrap the vectorizer + classifier into a single `sklearn.Pipeline`, which handles serialization more cleanly and makes cross-validation and hyperparameter tuning far simpler.

---

## Process Learnings

- **Document as you go**: Writing `Architecture.md` and `Data_Pipeline.md` early prevented scope creep and kept module boundaries clear.
- **Test edge cases first**: Empty PDFs, mismatched vector dimensions, and missing model files are the most common failure modes — write tests for these first.
- **Commit often, in small chunks**: Each module (pdf_extractor, text_cleaner, tokenizer) deserves its own commit, not one giant "add all modules" commit.
