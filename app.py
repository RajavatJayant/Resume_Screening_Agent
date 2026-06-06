"""
Resume Screening Agent — app.py
================================
MAIN FILE: This is where Streamlit UI lives.
Run with: streamlit run app.py

FLOW:
  Upload PDFs → Extract Text → Embed with SentenceTransformer
  → Store in FAISS → ATS Score → Rank → Gemini Analysis
"""

import streamlit as st
import os
from dotenv import load_dotenv

# ✅ Load .env FIRST before anything else
load_dotenv()

# Our custom modules
from resume_parser import extract_text_from_pdf
from vector_store import build_faiss_index, search_similar
from ats_scorer import calculate_ats_score
from ranker import rank_candidates
from report_generator import generate_report

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Resume Screening Agent")
st.caption("Powered by LangChain · FAISS · SentenceTransformers · Google Gemini")
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📋 Job Description")
    job_description = st.text_area(
        "Paste the Job Description here:",
        height=300,
        placeholder="e.g. We are looking for a Python developer with 2+ years experience in FastAPI, SQL, REST APIs..."
    )

    st.markdown("---")
    st.header("📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF resumes",
        type=["pdf"],
        accept_multiple_files=True
    )

    screen_button = st.button("🚀 Screen Resumes", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
if screen_button:

    if not job_description.strip():
        st.warning("⚠️ Please enter a job description.")
        st.stop()

    if not uploaded_files:
        st.warning("⚠️ Please upload at least one resume.")
        st.stop()

    # Check API key early
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found! Add it to your .env file.")
        st.stop()

    st.info(f"Processing {len(uploaded_files)} resume(s)... Please wait.")

    # ── STEP 1: Extract text from PDFs ──────────────
    st.subheader("📄 Step 1: Extracting Resume Text")
    resumes = {}

    for file in uploaded_files:
        candidate_name = file.name.replace(".pdf", "").replace("_", " ").replace("-", " ")
        text = extract_text_from_pdf(file)
        if text:
            resumes[candidate_name] = text
            st.success(f"✅ Extracted: {candidate_name}")
        else:
            st.error(f"❌ Could not read: {file.name} — make sure it's a text-based PDF")

    if not resumes:
        st.error("No readable resumes found.")
        st.stop()

    # ── STEP 2: Build FAISS vector index ────────────
    st.subheader("🧠 Step 2: Building Vector Index (SentenceTransformer + FAISS)")
    with st.spinner("Generating embeddings... this may take 30-60 seconds on first run."):
        index, resume_chunks = build_faiss_index(resumes)
    st.success(f"✅ Vector index built with {len(resume_chunks)} chunks.")

    # ── STEP 3: Search relevant sections ────────────
    st.subheader("🔍 Step 3: Matching Against Job Description")
    relevant_sections = search_similar(index, resume_chunks, job_description)
    st.success("✅ Similarity search complete.")

    # ── STEP 4: ATS Scoring ─────────────────────────
    st.subheader("📊 Step 4: Calculating ATS Scores")
    scores = {}

    for name, text in resumes.items():
        score, breakdown = calculate_ats_score(text, job_description)
        scores[name] = {
            "score": score,
            "breakdown": breakdown,
            "resume_text": text
        }
        st.write(f"**{name}**: `{score}/100`")

    # ── STEP 5: Rank candidates ──────────────────────
    ranked = rank_candidates(scores)

    # ── DISPLAY RESULTS ──────────────────────────────
    st.markdown("---")
    st.header("🏆 Final Rankings")

    for rank, (name, data) in enumerate(ranked, start=1):
        score = data["score"]
        breakdown = data["breakdown"]

        if score >= 75:
            badge = "🟢 Strong Match"
        elif score >= 50:
            badge = "🟡 Moderate Match"
        else:
            badge = "🔴 Weak Match"

        with st.expander(f"#{rank} — {name}  |  Score: {score}/100  |  {badge}"):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("ATS Score", f"{score}/100")
                st.write("**Score Breakdown:**")
                for category, cat_score in breakdown.items():
                    st.write(f"- {category}: `{cat_score}`")

            with col2:
                st.write("**AI Analysis (Gemini):**")
                with st.spinner("Generating analysis..."):
                    report = generate_report(name, data["resume_text"], job_description, score)
                st.write(report)

    # Download summary
    st.markdown("---")
    summary_lines = [
        f"Rank #{i+1}: {name} | Score: {data['score']}/100"
        for i, (name, data) in enumerate(ranked)
    ]
    st.download_button(
        "⬇️ Download Rankings Summary",
        data="\n".join(summary_lines),
        file_name="candidate_rankings.txt",
        mime="text/plain"
    )
