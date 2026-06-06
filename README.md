# 🤖 Resume Screening Agent

AI-powered resume screener using **SentenceTransformers + FAISS + LangChain + Google Gemini**.

---

## 📁 Project Structure

```
resume-screening-agent/
├── app.py                ← Streamlit UI (main entry point)
├── resume_parser.py      ← PDF → plain text (PyMuPDF)
├── vector_store.py       ← SentenceTransformer embeddings + FAISS index
├── ats_scorer.py         ← ATS score calculation (0-100)
├── ranker.py             ← Sort candidates by score
├── report_generator.py   ← LangChain + Gemini analysis
├── requirements.txt      ← All dependencies
├── .env.example          ← API key template
└── .gitignore            ← Prevents .env from being pushed
```

---

## 🚀 Local Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create your .env file
cp .env.example .env
# Edit .env → add your real Gemini API key
# Get free key: https://aistudio.google.com/app/apikey

# 3. Run the app
streamlit run app.py
```

---

## 🧠 Architecture (for interviews)

```
PDF Resumes (upload)
      ↓
[resume_parser.py]     → PyMuPDF extracts plain text
      ↓
[vector_store.py]      → SentenceTransformer encodes text into 384-dim vectors
                       → Chunks stored in FAISS IndexFlatL2
                       → Job Description also encoded
                       → FAISS search finds closest resume chunks
      ↓
[ats_scorer.py]        → 4-component score: Keywords(40) + Skills(30) + Exp(20) + Edu(10)
      ↓
[ranker.py]            → sorted() descending by score
      ↓
[report_generator.py]  → LangChain LCEL: PromptTemplate | Gemini | StrOutputParser
      ↓
[app.py Streamlit]     → Display ranked results with AI analysis
```

---

## 🎤 Interview Q&A

**Q: Why SentenceTransformer instead of Gemini embeddings?**
> SentenceTransformer runs locally — no API cost, no rate limits, and
> the `all-MiniLM-L6-v2` model is specifically trained for semantic
> similarity tasks which is exactly what resume matching needs.

**Q: What is FAISS?**
> FAISS (Facebook AI Similarity Search) is a library for fast nearest
> neighbour search in high-dimensional vector spaces. Instead of comparing
> every resume one by one, FAISS finds the closest vectors in milliseconds.

**Q: What is LangChain used for here?**
> LangChain simplifies working with LLMs. I used PromptTemplate to
> structure my prompt with variables, and LCEL (pipe syntax) to chain:
> prompt → Gemini model → output parser. This keeps the code clean and modular.

**Q: How is the ATS score calculated?**
> I broke it into 4 weighted components: keyword overlap with JD (40pts),
> technical skills match (30pts), experience keywords (20pts), and
> education keywords (10pts). Total = 100.

**Q: What model does Gemini use?**
> `gemini-1.5-flash` — it's fast, free-tier available, and good for
> short analysis/summarization tasks.
