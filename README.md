# AI Resume Screening & Skill Gap Analysis System

## Overview

The **AI Resume Screening & Skill Gap Analysis System** is a Machine Learning and Natural Language Processing (NLP) application that helps recruiters evaluate resumes against a given job description. The system automatically analyzes the uploaded resume, calculates a match score, identifies matching and missing skills, and provides a recommendation based on the candidate's suitability.

This project demonstrates how AI can simplify the recruitment process by reducing manual resume screening and providing objective candidate evaluation.

---

## Features

* Upload resume in PDF format
* Paste any job description
* Automatic resume text extraction
* Text preprocessing using NLP techniques
* Skill extraction from resume and job description
* Resume-to-job similarity scoring using TF-IDF and Cosine Similarity
* Skill gap identification
* Candidate recommendation
* Interactive Streamlit interface
* Visualizations of analysis results

---

## Technologies Used

### Programming Language

* Python

### Machine Learning & NLP

* Scikit-learn
* NLTK
* TF-IDF Vectorizer
* Cosine Similarity

### Libraries

* Pandas
* NumPy
* pdfplumber
* Joblib
* Matplotlib
* Plotly
* Streamlit

---

## Project Workflow

1. Upload a resume in PDF format.
2. Paste the job description.
3. Extract text from the uploaded resume.
4. Clean and preprocess the text using NLP.
5. Extract relevant skills from both resume and job description.
6. Convert text into TF-IDF vectors.
7. Compute cosine similarity between the resume and job description.
8. Calculate the overall match score.
9. Identify matched skills and missing skills.
10. Display the final recommendation along with visualizations.

---

## Project Structure

```text
AI_Resume_Screening/

├── app.py
├── dataset/
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── resume_vectors.pkl
│   └── resume_metadata.csv
├── notebooks/
│   └── training.ipynb
├── utils/
│   ├── pdf_reader.py
│   ├── preprocess.py
│   ├── predictor.py
│   └── skill_extractor.py
├── requirements.txt
└── README.md
```

---

## How It Works

The system accepts a resume in PDF format and a job description provided by the user. After extracting and preprocessing the text, the application converts both documents into TF-IDF vectors. Cosine Similarity is then used to measure how closely the resume matches the job description. Finally, the system identifies matching skills, highlights missing skills, calculates a match percentage, and generates a recommendation.

---

## Output

The application provides:

* Overall Match Score
* Candidate Recommendation
* Matched Skills
* Missing Skills
* Additional Skills
* Model Confidence
* Interactive Visualizations

---

## Applications

* Resume Screening
* Candidate Shortlisting
* Recruitment Automation
* HR Analytics
* Skill Gap Analysis
* Applicant Tracking Support

---

## Future Enhancements

* Support multiple resume uploads
* Automatic resume ranking
* Resume parsing using spaCy Named Entity Recognition
* AI-powered resume feedback
* Multi-job comparison
* Resume scoring dashboard
* Export reports as PDF
* Cloud deployment

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AI-Resume-Screening.git
```

Navigate to the project folder:

```bash
cd AI-Resume-Screening
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Screenshots

Add screenshots of:

* Home Screen
* Resume Upload
* Job Description Input
* Analysis Results
* Skill Gap Analysis
* Visualizations

---

## Author

**Vineela Lanka**

B.Tech – Computer Science and Engineering (AI & ML)

---

## License

This project is developed for educational and internship purposes.
