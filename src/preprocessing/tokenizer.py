"""
src.preprocessing.tokenizer
============================
Tokenizes cleaned text and applies NLP transformations:
    - Word tokenization via NLTK
    - Stopword removal
    - Lemmatization (WordNetLemmatizer)
    - Optional stemming (PorterStemmer)

Usage:
    from src.preprocessing.tokenizer import tokenize, tokens_to_string

    tokens = tokenize("machine learning engineer with python experience")
    processed_text = tokens_to_string(tokens)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-load NLTK resources to avoid startup errors when NLTK data is missing
_nltk_loaded: bool = False
_stopwords: Optional[set[str]] = None
_lemmatizer = None
_stemmer = None


def _ensure_nltk_resources() -> bool:
    """Download and load required NLTK resources if not already loaded.

    Returns:
        True if resources were loaded successfully, False otherwise.
    """
    global _nltk_loaded, _stopwords, _lemmatizer, _stemmer

    if _nltk_loaded:
        return True

    try:
        import nltk  # type: ignore[import]
        from nltk.corpus import stopwords as sw  # type: ignore[import]
        from nltk.stem import PorterStemmer, WordNetLemmatizer  # type: ignore[import]

        # Silently download if missing
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)

        _stopwords = set(sw.words("english"))
        _lemmatizer = WordNetLemmatizer()
        _stemmer = PorterStemmer()
        _nltk_loaded = True

        logger.debug("NLTK resources loaded successfully.")
        return True

    except ImportError:
        logger.error("NLTK is not installed. Run: pip install nltk")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load NLTK resources: %s", exc)
        return False


def tokenize(
    text: str,
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True,
    apply_stemming: bool = False,
    min_token_length: int = 2,
) -> list[str]:
    """Tokenize a cleaned text string into a list of processed tokens.

    Pipeline:
        text → word_tokenize → [remove_stopwords] → [lemmatize / stem]
                              → filter by min_token_length

    Args:
        text: Pre-cleaned lowercase text string.
        remove_stopwords: Whether to filter out NLTK English stopwords.
        apply_lemmatization: Whether to apply WordNet lemmatization.
        apply_stemming: Whether to apply Porter stemming (alternative to lemma).
                        Note: lemmatization and stemming are mutually exclusive;
                        lemmatization takes priority.
        min_token_length: Minimum character length for a token to be retained.

    Returns:
        List of processed token strings.

    Example:
        >>> tokenize("machine learning engineers develop systems")
        ['machine', 'learning', 'engineer', 'develop', 'system']
    """
    if not text or not text.strip():
        return []

    if not _ensure_nltk_resources():
        # Fallback: simple whitespace tokenization
        logger.warning("Using fallback whitespace tokenization.")
        return [t for t in text.split() if len(t) >= min_token_length]

    try:
        from nltk.tokenize import word_tokenize  # type: ignore[import]

        # Step 1: Tokenize
        tokens: list[str] = word_tokenize(text)

        # Step 2: Remove stopwords
        if remove_stopwords and _stopwords:
            tokens = [t for t in tokens if t not in _stopwords]

        # Step 3: Apply lemmatization (preferred over stemming)
        if apply_lemmatization and _lemmatizer:
            tokens = [_lemmatizer.lemmatize(t) for t in tokens]
        elif apply_stemming and _stemmer:
            tokens = [_stemmer.stem(t) for t in tokens]

        # Step 4: Filter by minimum token length
        tokens = [t for t in tokens if len(t) >= min_token_length]

        logger.debug("Tokenized to %d tokens.", len(tokens))
        return tokens

    except Exception as exc:  # noqa: BLE001
        logger.error("Tokenization failed: %s", exc)
        return []


def tokens_to_string(tokens: list[str], separator: str = " ") -> str:
    """Join a list of tokens back into a single string.

    Args:
        tokens: List of processed token strings.
        separator: String to insert between tokens.

    Returns:
        Single string of joined tokens.
    """
    return separator.join(tokens)


def get_stopwords() -> set[str]:
    """Return the set of English NLTK stopwords.

    Returns:
        Set of lowercase stopword strings, or empty set if NLTK unavailable.
    """
    _ensure_nltk_resources()
    return _stopwords or set()


# TODO: Implement custom domain-specific stopword list (e.g., "resume", "cv", "page")
# TODO: Add support for bigram and trigram extraction as post-tokenization step
