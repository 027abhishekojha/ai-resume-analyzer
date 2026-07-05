"""
src.preprocessing.pdf_extractor
================================
Extracts raw text content from PDF resume files.

Uses pdfplumber as the primary extractor and PyPDF2 as a fallback
to handle edge cases (scanned PDFs, complex layouts, etc.).

Usage:
    from src.preprocessing.pdf_extractor import extract_text_from_pdf

    text = extract_text_from_pdf("path/to/resume.pdf")
"""

import io
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def extract_text_from_pdf(
    file_source: Union[str, bytes, io.BytesIO],
    use_fallback: bool = True,
) -> str:
    """Extract plain text from a PDF file.

    Attempts extraction using pdfplumber first (handles complex layouts well).
    If pdfplumber fails or returns empty text, falls back to PyPDF2.

    Args:
        file_source: Path string, raw bytes, or BytesIO object of the PDF.
        use_fallback: Whether to attempt PyPDF2 if pdfplumber fails.

    Returns:
        Extracted text as a single string. Returns empty string on failure.

    Raises:
        ValueError: If the file_source type is not supported.
    """
    if isinstance(file_source, str):
        source_path = Path(file_source)
        if not source_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_source}")
        logger.info("Extracting text from PDF file: %s", file_source)
    elif isinstance(file_source, (bytes, io.BytesIO)):
        logger.info("Extracting text from in-memory PDF bytes.")
    else:
        raise ValueError(
            f"Unsupported file_source type: {type(file_source)}. "
            "Expected str, bytes, or BytesIO."
        )

    # Primary extraction: pdfplumber
    text = _extract_with_pdfplumber(file_source)

    if not text.strip() and use_fallback:
        logger.warning(
            "pdfplumber returned empty text. Attempting PyPDF2 fallback."
        )
        text = _extract_with_pypdf2(file_source)

    if not text.strip():
        logger.error(
            "Both extractors returned empty text. "
            "PDF may be scanned/image-based or corrupted."
        )

    logger.debug("Extracted %d characters from PDF.", len(text))
    return text


def _extract_with_pdfplumber(
    file_source: Union[str, bytes, io.BytesIO]
) -> str:
    """Extract text using the pdfplumber library.

    Args:
        file_source: Path string, bytes, or BytesIO of the PDF.

    Returns:
        Extracted text, or empty string if extraction fails.
    """
    try:
        import pdfplumber  # type: ignore[import]

        # Normalize source to a file-like object
        if isinstance(file_source, str):
            pdf_source: Union[str, io.BytesIO] = file_source
        elif isinstance(file_source, bytes):
            pdf_source = io.BytesIO(file_source)
        else:
            pdf_source = file_source

        pages_text: list[str] = []
        with pdfplumber.open(pdf_source) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
                    logger.debug(
                        "Page %d: extracted %d characters.",
                        page_num,
                        len(page_text),
                    )

        extracted = "\n".join(pages_text)
        logger.info(
            "pdfplumber extracted %d characters across %d pages.",
            len(extracted),
            len(pages_text),
        )
        return extracted

    except ImportError:
        logger.error("pdfplumber is not installed. Run: pip install pdfplumber")
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.error("pdfplumber extraction failed: %s", exc)
        return ""


def _extract_with_pypdf2(
    file_source: Union[str, bytes, io.BytesIO]
) -> str:
    """Extract text using the PyPDF2 library as a fallback.

    Args:
        file_source: Path string, bytes, or BytesIO of the PDF.

    Returns:
        Extracted text, or empty string if extraction fails.
    """
    try:
        import PyPDF2  # type: ignore[import]

        if isinstance(file_source, str):
            with open(file_source, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = _read_pypdf2_pages(reader)
        elif isinstance(file_source, bytes):
            reader = PyPDF2.PdfReader(io.BytesIO(file_source))
            text = _read_pypdf2_pages(reader)
        else:
            reader = PyPDF2.PdfReader(file_source)
            text = _read_pypdf2_pages(reader)

        logger.info("PyPDF2 extracted %d characters.", len(text))
        return text

    except ImportError:
        logger.error("PyPDF2 is not installed. Run: pip install PyPDF2")
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.error("PyPDF2 extraction failed: %s", exc)
        return ""


def _read_pypdf2_pages(reader: "PyPDF2.PdfReader") -> str:  # type: ignore[name-defined]
    """Extract and concatenate text from all pages of a PyPDF2 reader.

    Args:
        reader: An initialized PyPDF2.PdfReader instance.

    Returns:
        Concatenated text from all pages.
    """
    pages_text: list[str] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
            logger.debug("Page %d: %d characters.", page_num, len(page_text))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read page %d: %s", page_num, exc)
    return "\n".join(pages_text)


# TODO: Add OCR support via pytesseract for scanned/image-based PDFs
# TODO: Add support for extracting tables (pdfplumber has native table support)
# TODO: Add support for DOCX files via python-docx
