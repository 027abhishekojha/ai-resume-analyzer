# 🤝 Contributing to AI Resume Analyzer

Thank you for your interest in contributing! We welcome contributions of all kinds — bug reports, feature requests, documentation improvements, and code changes.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Branch Strategy](#branch-strategy)
- [Commit Conventions](#commit-conventions)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-resume-analyzer.git
   cd ai-resume-analyzer
   ```
3. **Add upstream** remote:
   ```bash
   git remote add upstream https://github.com/yourusername/ai-resume-analyzer.git
   ```

---

## Development Setup

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install all dependencies including dev tools
pip install -r requirements.txt
pip install -e ".[dev]"

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy environment file
cp .env.example .env
```

---

## Branch Strategy

We follow a **GitFlow-inspired** branching model:

| Branch | Purpose |
|---|---|
| `main` | Production-ready, always stable |
| `develop` | Integration branch for features |
| `feature/*` | New features (e.g., `feature/bert-embeddings`) |
| `bugfix/*` | Bug fixes (e.g., `bugfix/pdf-parse-error`) |
| `release/*` | Release preparation (e.g., `release/v0.2.0`) |
| `hotfix/*` | Urgent production fixes |

### Workflow

```bash
# Always branch from develop
git checkout develop
git pull upstream develop
git checkout -b feature/your-feature-name

# After completing work
git push origin feature/your-feature-name
# Open PR targeting develop
```

---

## Commit Conventions

We use **Conventional Commits** for clear, automated changelogs:

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

### Types

| Type | When to Use |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `style` | Formatting, no logic changes |
| `refactor` | Code restructuring, no new features |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks, dependency updates |
| `perf` | Performance improvements |

### Examples

```bash
git commit -m "feat(inference): add cosine similarity scoring engine"
git commit -m "fix(preprocessing): handle empty PDF text gracefully"
git commit -m "docs(readme): add installation instructions for Windows"
git commit -m "test(preprocessing): add unit tests for text cleaner"
```

---

## Pull Request Process

1. **Ensure tests pass**: `pytest tests/ -v`
2. **Ensure formatting is correct**: `black src/ tests/`
3. **Ensure linting passes**: `flake8 src/ tests/`
4. **Update documentation** if you change APIs or behavior
5. **Update CHANGELOG.md** under `[Unreleased]`
6. **Fill in the PR template** completely
7. **Request review** from at least one maintainer

PRs will be merged only after:
- ✅ All CI checks pass
- ✅ At least one approving review
- ✅ No unresolved conversations

---

## Coding Standards

### Python Style
- Follow **PEP 8** strictly
- Use **Black** for formatting (line length: 88)
- Use **Flake8** for linting
- Use **type hints** on all function signatures
- Write **docstrings** for all public functions, classes, and modules (Google style)

### Example

```python
def compute_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two dense vectors.

    Args:
        vec_a: First feature vector of shape (n_features,).
        vec_b: Second feature vector of shape (n_features,).

    Returns:
        Cosine similarity score in range [0.0, 1.0].

    Raises:
        ValueError: If input vectors have mismatched shapes.
    """
    # Implementation here
    ...
```

---

## Testing Guidelines

- Place tests in `tests/` mirroring the `src/` structure
- Name test files `test_<module>.py`
- Name test functions `test_<what_is_being_tested>`
- Aim for **>80% code coverage** on new modules
- Use **pytest fixtures** for reusable test data

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

---

## Reporting Issues

When reporting a bug, please include:

- **Python version** and OS
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Error messages** or tracebacks (full text)
- **Sample files** (anonymized resume or JD) if applicable

Use the [GitHub Issues](https://github.com/yourusername/ai-resume-analyzer/issues) page and apply appropriate labels.

---

Thank you for contributing! 🎉
