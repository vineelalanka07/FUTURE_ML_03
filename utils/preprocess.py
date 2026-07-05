"""Text preprocessing helpers for the Streamlit frontend.

These functions are currently placeholders and should be wired to the
trained preprocessing pipeline once the backend is connected.
"""


def preprocess_text(text: str) -> str:
    """Return cleaned text for downstream matching."""
    if not text:
        return ""
    return " ".join(text.lower().split())
