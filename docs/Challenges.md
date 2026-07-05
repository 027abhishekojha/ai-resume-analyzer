# 🧗 Challenges — AI Resume Analyzer

## 1. PDF Parsing Heterogeneity

**Challenge**: Resumes come in an enormous variety of layouts — single column, multi-column, tables, text boxes, embedded graphics, and non-selectable scanned images. No single PDF library handles all cases perfectly.

**Approach**: Used pdfplumber as the primary extractor (superior for structured layouts) with a PyPDF2 fallback. For scanned PDFs, OCR via pytesseract is planned.

**Lesson**: Always build fallback strategies into I/O-heavy pipelines.

---

## 2. Absence of Labeled Training Data

**Challenge**: Building a supervised classifier requires thousands of labeled (resume, job_description, label) pairs with verified ground truth. Such datasets are either proprietary (LinkedIn, Indeed) or small/noisy when public.

**Approach**: Used a synthetic data generator for initial development. Planning to collect public Kaggle datasets and crowdsource annotation.

**Lesson**: Data quality and availability are often the hardest ML problems — invest early.

---

## 3. TF-IDF Semantic Limitations

**Challenge**: TF-IDF is a bag-of-words model. It cannot understand that "ML engineer" and "machine learning engineer" are equivalent, or that "proficient in Python" maps to "Python skills required."

**Approach**: Applied lemmatization to reduce surface variation. The long-term fix is to replace TF-IDF with sentence embeddings (BERT, sentence-transformers).

**Lesson**: Always validate whether your feature representation actually captures the semantic relationships that matter for your task.

---

## 4. Resume Section Noise

**Challenge**: Structural headers ("EDUCATION", "SKILLS", "REFERENCES") and filler phrases ("References available upon request") inflate TF-IDF vocabulary without adding signal.

**Approach**: Built a `remove_resume_headers()` utility that strips common section labels before tokenization.

---

## 5. ATS System Variability

**Challenge**: Different Applicant Tracking Systems (Taleo, Workday, Greenhouse) apply wildly different parsing and scoring logic. There is no universal ATS standard to replicate.

**Approach**: Focused on building a generally useful resume-JD similarity tool rather than mimicking any specific ATS. Made all scoring thresholds configurable so users can tune to their context.

---

## 6. Class Imbalance

**Challenge**: In real-world data, "Not Suitable" candidates vastly outnumber "Suitable" ones (often 10:1 or worse). A naive classifier will simply predict "Not Suitable" for everything and achieve 90%+ accuracy while being useless.

**Approach**: The training pipeline includes provisions for class weighting and plans to add SMOTE-based oversampling.
