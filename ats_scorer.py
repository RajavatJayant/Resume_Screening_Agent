"""
ats_scorer.py
=============
PURPOSE: Score each resume against the job description (0-100 points).

WHAT IS ATS?
  Applicant Tracking System — software companies use to auto-filter resumes.
  Resumes with low keyword match get rejected before a human even reads them.

OUR SCORING BREAKDOWN (Total = 100):
  1. Keyword Match   → 40 pts  (most important — direct word overlap)
  2. Skills Match    → 30 pts  (technical skills from job description)
  3. Experience      → 20 pts  (experience-related words in resume)
  4. Education       → 10 pts  (degree/qualification keywords)

INTERVIEW EXPLANATION:
"I broke the ATS score into 4 weighted categories.
Keyword match is highest because ATS systems primarily scan for
keywords from the job description. I compute overlap between
resume words and JD words, then scale to a 0-100 score."
"""

import re


# ── Common technical skills to scan for ──────────────────────────────────────
TECH_SKILLS = [
    "python", "java", "javascript", "typescript", "react", "node", "nodejs",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "fastapi", "django", "flask", "spring", "express",
    "docker", "kubernetes", "aws", "gcp", "azure", "terraform", "ci/cd",
    "git", "github", "gitlab", "linux", "bash",
    "machine learning", "deep learning", "nlp", "langchain", "llm",
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
    "rest api", "graphql", "microservices", "kafka", "spark",
    "html", "css", "vue", "angular", "tailwind",
    "c++", "c#", "golang", "rust", "kotlin", "swift",
    "data structures", "algorithms", "system design", "oops"
]

# ── Education keywords ────────────────────────────────────────────────────────
EDUCATION_KEYWORDS = [
    "b.tech", "btech", "b.e", "bachelor", "b.sc", "bsc", "b.com",
    "m.tech", "mtech", "m.sc", "msc", "mba", "master", "phd", "diploma",
    "computer science", "information technology", "engineering",
    "university", "college", "institute", "degree", "cgpa", "gpa"
]

# ── Experience keywords ───────────────────────────────────────────────────────
EXPERIENCE_KEYWORDS = [
    "years", "experience", "worked", "developed", "built", "designed",
    "implemented", "led", "managed", "created", "deployed", "maintained",
    "intern", "internship", "project", "team", "collaborated", "delivered",
    "optimized", "automated", "integrated", "architected", "responsible"
]

# ── Stop words (not useful for matching) ─────────────────────────────────────
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "we", "you", "our", "your",
    "they", "their", "it", "its", "as", "if", "then", "than", "so", "not"
}


def clean_and_tokenize(text: str) -> set:
    """Lowercase, remove punctuation, remove stop words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 2}


def score_keyword_match(resume_text: str, jd_text: str) -> int:
    """40 pts — how many JD words appear in the resume."""
    resume_words = clean_and_tokenize(resume_text)
    jd_words = clean_and_tokenize(jd_text)
    if not jd_words:
        return 0
    overlap = resume_words & jd_words
    ratio = len(overlap) / len(jd_words)
    return min(int(ratio * 40), 40)


def score_skills_match(resume_text: str, jd_text: str) -> int:
    """30 pts — how many required tech skills does the candidate have."""
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()

    required = [s for s in TECH_SKILLS if s in jd_lower]
    if not required:
        return 15  # Partial score if JD has no clear tech skills listed

    matched = [s for s in required if s in resume_lower]
    ratio = len(matched) / len(required)
    return min(int(ratio * 30), 30)


def score_experience(resume_text: str) -> int:
    """20 pts — does the resume show work experience."""
    resume_lower = resume_text.lower()
    found = sum(1 for kw in EXPERIENCE_KEYWORDS if kw in resume_lower)
    return min(int((found / 8) * 20), 20)


def score_education(resume_text: str) -> int:
    """10 pts — does the resume mention education qualifications."""
    resume_lower = resume_text.lower()
    found = sum(1 for kw in EDUCATION_KEYWORDS if kw in resume_lower)
    if found >= 3:
        return 10
    elif found >= 1:
        return 5
    return 0


def calculate_ats_score(resume_text: str, jd_text: str) -> tuple:
    """
    Main function — returns (total_score, breakdown_dict).

    Args:
        resume_text: Full plain text of the resume
        jd_text    : Job description text

    Returns:
        (int score 0-100, dict breakdown)
    """
    kw   = score_keyword_match(resume_text, jd_text)
    sk   = score_skills_match(resume_text, jd_text)
    exp  = score_experience(resume_text)
    edu  = score_education(resume_text)

    total = kw + sk + exp + edu

    breakdown = {
        "🔑 Keyword Match  (max 40)": kw,
        "💻 Skills Match   (max 30)": sk,
        "💼 Experience     (max 20)": exp,
        "🎓 Education      (max 10)": edu,
    }

    return total, breakdown
