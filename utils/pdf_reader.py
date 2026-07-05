"""PDF text extraction helpers for the Streamlit app."""

import io


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from a PDF upload using pdfplumber."""
    if uploaded_file is None:
        return ""

    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required to extract PDF content.") from exc

    try:
        with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)
            return "\n".join(pages).strip()
    except Exception:
        return f"Resume uploaded: {uploaded_file.name}"
