"""
report_generator.py
===================
PURPOSE: Use Gemini LLM via LangChain to write a short analysis per candidate.

LANGCHAIN CONCEPTS USED (explain in interview):
  - ChatGoogleGenerativeAI : LangChain wrapper around Gemini
  - PromptTemplate         : Reusable prompt with {variables}
  - StrOutputParser        : Converts LLM output to plain string
  - Chain (|)              : Connects prompt → model → parser in one pipeline

NEW LANGCHAIN v2 SYNTAX (LangChain Expression Language = LCEL):
  chain = prompt | llm | parser
  response = chain.invoke({"var": "value"})

INTERVIEW EXPLANATION:
"LangChain is a framework that simplifies working with LLMs.
I used PromptTemplate to define a structured prompt with variables,
then chained it to Gemini using LCEL (pipe syntax).
The StrOutputParser converts the LLM's response object to a plain string."
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env FIRST

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_llm():
    """
    Returns Gemini 1.5 Flash model via LangChain.

    WHY gemini-1.5-flash?
    - Fast and free-tier friendly
    - Good for short analysis tasks
    - gemini-pro is deprecated in newer SDK versions
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found!\n"
            "Create a .env file in your project folder with:\n"
            "GOOGLE_API_KEY=your_key_here\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )

    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",   # ✅ Current working free model
        google_api_key=api_key,
        temperature=0.3             # Low = factual; High = creative
    )


def generate_report(candidate_name: str, resume_text: str,
                    job_description: str, ats_score: int) -> str:
    """
    Generates a short AI analysis of the candidate using Gemini.

    LangChain LCEL chain:
      PromptTemplate → fills variables
      ChatGoogleGenerativeAI → sends to Gemini
      StrOutputParser → returns plain string

    Args:
        candidate_name : Name shown in UI
        resume_text    : Full resume text (we only send first 1500 chars)
        job_description: The job description
        ats_score      : The calculated score

    Returns:
        str: 3-4 line analysis from Gemini
    """

    # ── Define prompt template ────────────────────────────────────────────────
    # {variable_name} placeholders get filled by .invoke()
    prompt = PromptTemplate(
        input_variables=["candidate_name", "ats_score", "job_description", "resume_text"],
        template="""
You are a professional HR recruiter reviewing resumes.

Candidate: {candidate_name}
ATS Score: {ats_score}/100

Job Description (summary):
{job_description}

Resume (excerpt):
{resume_text}

Write a SHORT 3-4 sentence professional analysis:
1. Key strengths that match the job
2. Any skill gaps or weaknesses
3. Final recommendation: Shortlist / Consider / Reject

Be concise and professional.
"""
    )

    try:
        llm = get_llm()

        # ── LangChain LCEL chain (pipe syntax) ───────────────────────────────
        # prompt | llm | parser is the new LangChain v2 way
        chain = prompt | llm | StrOutputParser()

        # ── Run the chain ─────────────────────────────────────────────────────
        response = chain.invoke({
            "candidate_name": candidate_name,
            "ats_score": ats_score,
            "job_description": job_description[:600],   # Limit tokens
            "resume_text": resume_text[:1500]           # Limit tokens
        })

        return response.strip()

    except Exception as e:
        # Fallback if Gemini fails — still useful output
        print(f"[report_generator] Gemini error: {e}")
        if ats_score >= 75:
            return (f"✅ **Recommendation: Shortlist**\n"
                    f"{candidate_name} shows strong alignment with the job requirements. "
                    f"ATS score of {ats_score}/100 indicates good keyword and skills match. "
                    f"Recommend proceeding to interview stage.")
        elif ats_score >= 50:
            return (f"🟡 **Recommendation: Consider**\n"
                    f"{candidate_name} partially matches the requirements with a score of {ats_score}/100. "
                    f"Some skill gaps may exist. Consider a screening call to assess fit.")
        else:
            return (f"🔴 **Recommendation: Reject**\n"
                    f"{candidate_name} scored {ats_score}/100, indicating limited match "
                    f"with the job requirements. Does not meet minimum criteria.")
