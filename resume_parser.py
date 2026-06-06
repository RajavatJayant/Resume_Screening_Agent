"""
resume_parser.py
================
PURPOSE: Extract plain text from uploaded PDF resumes.

LIBRARY: PyMuPDF (imported as fitz)
- Fast and reliable PDF reader
- Works with text-based PDFs (not scanned images)

INTERVIEW EXPLANATION:
"AI models work with text, not PDFs directly.
So Step 1 is always: convert PDF → plain text string."
"""

import fitz  # PyMuPDF  →  pip install pymupdf


def extract_text_from_pdf(pdf_file) -> str:
    """
    Reads a Streamlit-uploaded PDF and returns all text as one string.

    Args:
        pdf_file: UploadedFile object from st.file_uploader

    Returns:
        str: Full text of the PDF, or "" if unreadable
    """
    try:
        pdf_bytes = pdf_file.read()                          # Read raw bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")   # Open from memory

        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"             # Extract each page

        doc.close()
        return full_text.strip()

    except Exception as e:
        print(f"[resume_parser] Error reading {pdf_file.name}: {e}")
        return ""
