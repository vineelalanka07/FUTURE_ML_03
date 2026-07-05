"""Skill extraction placeholders for the resume screening app."""


def extract_skills(text: str):
    """Return a list of skills derived from text.

    Replace this with the production skill extraction logic that uses the
    trained NLP pipeline from the models/ directory.
    """
    if not text:
        return []

    skills = ["Python", "SQL", "Machine Learning", "Git"]
    return skills
