"""Prediction wrapper for the Streamlit frontend."""

import os
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
RESUME_VECTORS_PATH = MODELS_DIR / "resume_vectors.pkl"
METADATA_PATH = MODELS_DIR / "resume_metadata.csv"

STOP_WORDS = None
LEMMATIZER = None
SKILL_CATALOG = (
    "python",
    "go",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "data analysis",
    "pandas",
    "numpy",
    "tensorflow",
    "pytorch",
    "jax",
    "git",
    "docker",
    "aws",
    "kubernetes",
    "cloud computing",
    "api development",
    "communication",
    "project management",
    "javascript",
    "react",
    "django",
    "flask",
    "statistics",
    "tableau",
    "power bi",
    "linux",
    "devops",
    "natural language processing",
    "nlp",
    "computer vision",
    "apache spark",
    "hadoop",
    "snowflake",
    "mlops",
    "production deployment",
    "system architecture",
    "code review",
    "mentoring",
    "technical leadership",
)


def _ensure_nltk_resources():
    global STOP_WORDS, LEMMATIZER
    if STOP_WORDS is not None and LEMMATIZER is not None:
        return
    try:
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()


def preprocess_text(text):
    _ensure_nltk_resources()
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    # Keep alphanumeric, spaces, and common technical characters
    text = re.sub(r"[^a-z0-9\s\-+.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = wordpunct_tokenize(text)
    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]
    return " ".join(tokens)


def _load_artifacts():
    if not VECTORIZER_PATH.exists() or not RESUME_VECTORS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Model artifacts were not found in the models/ directory.")
    vectorizer = joblib.load(VECTORIZER_PATH)
    resume_vectors = joblib.load(RESUME_VECTORS_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    return vectorizer, resume_vectors, metadata


def extract_skills(text):
    """Extract skills from text by matching against the skill catalog.
    
    Uses flexible substring matching for better recall.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    text_lower = text.lower()
    # Normalize whitespace and common punctuation for matching
    text_normalized = re.sub(r"[,\-+()/'\"]+", " ", text_lower)
    text_normalized = re.sub(r"\s+", " ", text_normalized)
    
    skills = []
    matched_skills_set = set()  # To avoid duplicates
    
    for skill in SKILL_CATALOG:
        skill_lower = skill.lower()
        
        # Try matching in both original and normalized text
        # For multi-word skills
        if " " in skill_lower:
            if skill_lower in text_lower or skill_lower in text_normalized:
                matched_skills_set.add(skill.title())
            # Also try with alternative spacing
            skill_spaced = " ".join(skill_lower.split())
            if skill_spaced in text_normalized:
                matched_skills_set.add(skill.title())
        # For single-word skills
        else:
            # Simple substring matching for skills with special characters
            if "+" in skill_lower or "-" in skill_lower:
                if skill_lower in text_lower:
                    matched_skills_set.add(skill.title())
            # For regular single words, be a bit more flexible
            else:
                # Check in both original and normalized text
                if re.search(rf"(^|\s|,|-){re.escape(skill_lower)}(\s|,|$|-|\.)", " " + text_normalized + " "):
                    matched_skills_set.add(skill.title())
    
    return list(matched_skills_set)


def analyze_resume(uploaded_file, resume_text, job_description):
    """Return a structured analysis result for the UI using the trained model artifacts."""
    vectorizer, resume_vectors, metadata = _load_artifacts()

    cleaned_resume_text = preprocess_text(resume_text)
    cleaned_job_text = preprocess_text(job_description)

    resume_vector = vectorizer.transform([cleaned_resume_text])
    job_vector = vectorizer.transform([cleaned_job_text])

    # Fallback to skill-based matching if TF-IDF vectors are empty
    if resume_vector.nnz == 0 or job_vector.nnz == 0:
        job_skills = extract_skills(job_description)
        resume_skills = extract_skills(resume_text)
        
        # If we couldn't extract ANY skills from either doc, try a more lenient approach
        if not job_skills and not resume_skills:
            # Use simple keyword matching as last resort
            job_keywords = set(job_description.lower().split())
            resume_keywords = set(resume_text.lower().split())
            common_keywords = job_keywords & resume_keywords
            
            if len(common_keywords) < 3:
                raise ValueError("Could not extract sufficient information from resume or job description. Ensure PDF is readable and contains text.")
            
            # Provide a basic score based on common words
            match_score = min(99.9, (len(common_keywords) / len(job_keywords) * 100)) if job_keywords else 0.0
            match_score = round(float(match_score), 2)
            recommendation = "Recommended" if match_score >= 50 else "Needs Improvement"
            
            return {
                "match_score": match_score,
                "recommendation": recommendation,
                "matched_skills": [],
                "missing_skills": [],
                "additional_skills": [],
                "confidence": 45.0,
                "extracted_resume_text": resume_text,
                "cleaned_resume_text": cleaned_resume_text,
                "skill_status": [],
            }
        
        matched_skills = [s for s in job_skills if s.lower() in [r.lower() for r in resume_skills]]
        missing_skills = [s for s in job_skills if s.lower() not in [r.lower() for r in resume_skills]]
        additional_skills = [s for s in resume_skills if s.lower() not in [j.lower() for j in job_skills]]
        
        # Calculate match score based on skill overlap
        match_score = (len(matched_skills) / len(job_skills) * 100) if job_skills else 0.0
        match_score = round(min(99.9, match_score), 2)
        
        if match_score >= 80:
            recommendation = "Highly Recommended"
        elif match_score >= 60:
            recommendation = "Recommended"
        else:
            recommendation = "Needs Improvement"
        
        return {
            "match_score": match_score,
            "recommendation": recommendation,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "additional_skills": additional_skills,
            "confidence": round(min(99.9, max(0.0, match_score)), 1),
            "extracted_resume_text": resume_text,
            "cleaned_resume_text": cleaned_resume_text,
            "skill_status": [(skill, "Matched") for skill in matched_skills] + [(skill, "Missing") for skill in missing_skills],
        }

    score = cosine_similarity(job_vector, resume_vector).ravel()[0] * 100
    match_score = round(float(score), 2)

    matched_skills = [
        skill for skill in extract_skills(job_description) if skill.lower() in cleaned_resume_text
    ]
    missing_skills = [
        skill for skill in extract_skills(job_description) if skill.lower() not in cleaned_resume_text
    ]
    additional_skills = [
        skill for skill in extract_skills(resume_text) if skill.lower() not in {s.lower() for s in matched_skills}
    ]

    if match_score >= 80:
        recommendation = "Highly Recommended"
    elif match_score >= 60:
        recommendation = "Recommended"
    else:
        recommendation = "Needs Improvement"

    return {
        "match_score": match_score,
        "recommendation": recommendation,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "additional_skills": additional_skills,
        "confidence": round(min(99.9, max(0.0, match_score)), 1),
        "extracted_resume_text": resume_text,
        "cleaned_resume_text": cleaned_resume_text,
        "skill_status": [(skill, "Matched") for skill in matched_skills] + [(skill, "Missing") for skill in missing_skills],
    }
