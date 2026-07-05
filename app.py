import streamlit as st
import matplotlib.pyplot as plt

from utils.predictor import analyze_resume
from utils.pdf_reader import extract_text_from_pdf

st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None


st.title("AI Resume Screening & Skill Gap Analysis")
st.caption("Upload a resume PDF and paste the job description to evaluate candidate suitability.")

with st.sidebar:
    st.header("Project Information")
    st.markdown("**Model**\nTF-IDF + Cosine Similarity")
    st.markdown("**Framework**\nStreamlit")
    st.markdown("**NLP**\nNLTK")
    st.markdown("**Vectorizer**\nTF-IDF")

    st.divider()
    st.caption("The trained artifacts are expected under the models/ folder.")


resume_tab, results_tab = st.tabs(["📄 Resume Screening", "📊 Results & Visualizations"])

with resume_tab:
    st.subheader("Resume Upload")
    uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])

    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the complete job description here...",
        placeholder="Paste the complete job description here...",
        height=220,
    )

    if st.button("Analyze Resume", width="stretch"):
        if uploaded_file is None or not job_description.strip():
            st.warning("Please upload a resume PDF and provide a job description.")
        else:
            with st.spinner("Analyzing resume and matching skills..."):
                # Run the prediction and capture errors to surface in the UI.
                try:
                    resume_text = extract_text_from_pdf(uploaded_file)
                    result = analyze_resume(
                        uploaded_file=uploaded_file,
                        resume_text=resume_text,
                        job_description=job_description,
                    )
                    st.session_state.analysis_result = result
                except Exception as exc:
                    st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
                    # keep previous result if present
                    st.session_state.analysis_result = None

    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        st.divider()
        st.subheader("Prediction Results")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Overall Match Score", f"{result['match_score']:.2f}%")
        with metric_col2:
            st.metric("Recommendation", result["recommendation"])
        with metric_col3:
            st.metric("Model Confidence", f"{result['confidence']:.1f}%")

        skill_cols = st.columns(3)
        with skill_cols[0]:
            st.markdown("### Matched Skills")
            for skill in result["matched_skills"]:
                st.markdown(f"<span style='background:#D1FAE5;color:#065F46;padding:4px 10px;border-radius:999px;display:inline-block;margin:2px;'>#{skill}</span>", unsafe_allow_html=True)
        with skill_cols[1]:
            st.markdown("### Missing Skills")
            for skill in result["missing_skills"]:
                st.markdown(f"<span style='background:#FEE2E2;color:#991B1B;padding:4px 10px;border-radius:999px;display:inline-block;margin:2px;'>#{skill}</span>", unsafe_allow_html=True)
        with skill_cols[2]:
            st.markdown("### Additional Skills")
            for skill in result["additional_skills"]:
                st.markdown(f"<span style='background:#DBEAFE;color:#1D4ED8;padding:4px 10px;border-radius:999px;display:inline-block;margin:2px;'>#{skill}</span>", unsafe_allow_html=True)

        # Debug: show the raw prediction JSON and extracted/cleaned texts for troubleshooting
        with st.expander("Raw Prediction JSON (debug)"):
            try:
                st.json(result)
            except Exception:
                st.write(result)

        with st.expander("Debug: Extracted & Cleaned Texts"):
            st.write("**Extracted resume text (first 4000 chars):**")
            st.write(result.get("extracted_resume_text", ""))
            st.write("---")
            st.write("**Cleaned resume text:**")
            st.write(result.get("cleaned_resume_text", ""))

with results_tab:
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        st.metric("Match Score", f"{result['match_score']:.2f}%")
        st.progress(result["match_score"] / 100)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            labels = ["Matched", "Missing"]
            sizes = [len(result["matched_skills"]), len(result["missing_skills"])]
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=["#10B981", "#EF4444"], startangle=90)
            ax.set_title("Skill Coverage")
            st.pyplot(fig)

        with chart_col2:
            fig, ax = plt.subplots(figsize=(4, 3))
            categories = ["Matched", "Missing"]
            values = [len(result["matched_skills"]), len(result["missing_skills"])]
            ax.barh(categories, values, color=["#10B981", "#EF4444"])
            ax.set_title("Matched vs Missing Skills")
            st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        top_skills = result["matched_skills"][:5]
        if len(top_skills) > 0:
            values = list(range(max(1, len(top_skills)), 0, -1))
            ax.bar(top_skills, values, color="#3B82F6")
        else:
            ax.text(0.5, 0.5, "No matched skills", ha="center", va="center")
            ax.set_xticks([])
        ax.set_title("Top Skills Found")
        ax.set_ylabel("Relevance")
        plt.xticks(rotation=20)
        st.pyplot(fig)

        with st.expander("Extracted Resume Text"):
            st.write(result["extracted_resume_text"])

        with st.expander("Cleaned Resume Text"):
            st.write(result["cleaned_resume_text"])

        st.subheader("Skill Status Table")
        skill_df = []
        for skill, status in result["skill_status"]:
            skill_df.append({"Skill": skill, "Status": status})
        st.dataframe(skill_df, width="stretch")
    else:
        st.info("Run an analysis from the Resume Screening tab to populate these visualizations.")
