import os
import json
import re
import time
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from google import genai


# ============================
# Safety + positioning
# ============================
DISCLAIMER = """
**Study tool only (not medical advice).**
This app generates practice questions for learning and exam prep.
It is not a substitute for clinical judgment, supervision, or official guidelines.
Do not use it to make real patient-care decisions.
"""

# ============================
# Config
# ============================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Pick a model that works for your account.
# If you see "model not found", change this string to the one that works for you.
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

MAX_RETRIES = 4


# ============================
# Gemini call wrapper (rate limit backoff)
# ============================
def _parse_retry_seconds(err_text: str) -> Optional[float]:
    # Examples: "Please retry in 23.54s", "retryDelay': '23s'"
    m = re.search(r"Please retry in\s+([0-9]+(\.[0-9]+)?)s", err_text)
    if m:
        return float(m.group(1))
    m2 = re.search(r"retryDelay'\s*:\s*'([0-9]+)s'", err_text)
    if m2:
        return float(m2.group(1))
    return None


def generate_with_backoff(prompt: str) -> str:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "Quota exceeded" in msg:
                wait_s = _parse_retry_seconds(msg)
                if wait_s is None:
                    wait_s = min(30.0, 2.0 ** attempt)
                time.sleep(wait_s)
                continue
            raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_err}")


# ============================
# MCQ generation
# ============================
def build_mcq_prompt(
    level: str,
    num_questions: int,
    topic: str,
    source_text: str,
    difficulty: str,
) -> str:
    """
    Single-shot generation (1 call) to reduce quota usage.
    Returns JSON so we can reliably render the quiz UI.
    """
    # Level guidance
    level_rules = {
        "Med School": "Focus on foundational IM concepts. Straightforward stems. Emphasize pathophysiology, basic diagnostics, and first-line management.",
        "Residency": "More clinical reasoning. Include next-best-step, differential diagnosis, initial workup, and guideline-consistent management. Avoid obscure zebras unless clearly signposted.",
        "Royal College": "Exam-style nuance. Include pitfalls, contraindications, complications, and management under constraints. Expect justification-level reasoning but still MCQ format.",
    }

    return f"""
You are generating exam-style practice questions for **Internal Medicine**.

LEVEL:
{level} — {level_rules.get(level, "")}

TOPIC FOCUS:
{topic}

DIFFICULTY:
{difficulty}

SOURCE MATERIAL (use if relevant; if empty, rely on standard IM knowledge):
{source_text if source_text.strip() else "[No source text provided]"}

TASK:
Create {num_questions} high-quality MCQs.

RULES:
- Each question must have 4 options labeled A, B, C, D.
- Provide exactly one correct answer.
- Provide a short explanation of why the correct answer is correct AND why one tempting distractor is wrong.
- Do NOT include any patient-identifying data.
- Keep this as study content; do not give real-world medical advice.

OUTPUT FORMAT:
Return VALID JSON ONLY, no markdown, no extra text.
JSON schema:
{{
  "questions": [
    {{
      "id": 1,
      "stem": "question text",
      "choices": {{"A":"...", "B":"...", "C":"...", "D":"..."}},
      "answer": "A",
      "explanation": "why correct + why a distractor is wrong"
    }}
  ]
}}
""".strip()


def extract_json(text: str) -> Dict[str, Any]:
    """
    Gemini sometimes returns extra text; try to extract JSON object safely.
    """
    text = text.strip()

    # If it's already valid JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("Could not parse JSON from model output.")


@st.cache_data(show_spinner=False)
def generate_mcqs_cached(
    level: str,
    num_questions: int,
    topic: str,
    source_text: str,
    difficulty: str,
) -> Dict[str, Any]:
    prompt = build_mcq_prompt(level, num_questions, topic, source_text, difficulty)
    raw = generate_with_backoff(prompt)
    return extract_json(raw)


# ============================
# UI
# ============================
st.set_page_config(page_title="IM Study MCQ Prototype", layout="wide")
st.title("IM Study MCQ Prototype")
st.markdown(DISCLAIMER)

with st.sidebar:
    st.header("Settings")
    level = st.selectbox("Learner level", ["Med School", "Residency", "Royal College"])
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    topic = st.text_input("Topic (e.g., CHF, COPD, AKI, PE, Diabetes)", value="Acute Coronary Syndrome")
    num_questions = st.slider("Number of questions", 5, 20, 10)

    st.divider()
    st.subheader("Optional source notes (TXT)")
    uploaded = st.file_uploader("Upload a .txt file (optional)", type=["txt"])
    source_text = ""
    if uploaded is not None:
        source_text = uploaded.read().decode("utf-8", errors="ignore")

    st.divider()
    st.caption("Tip: For free-tier quotas, keep question sets small (5–10).")

# Session state
if "mcq_data" not in st.session_state:
    st.session_state.mcq_data = None
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "submitted" not in st.session_state:
    st.session_state.submitted = False

col1, col2 = st.columns([1, 1], vertical_alignment="top")

with col1:
    st.subheader("Generate")
    if st.button("Generate MCQs", type="primary"):
        st.session_state.submitted = False
        st.session_state.answers = {}
        with st.spinner("Generating questions..."):
            try:
                st.session_state.mcq_data = generate_mcqs_cached(
                    level=level,
                    num_questions=num_questions,
                    topic=topic,
                    source_text=source_text,
                    difficulty=difficulty,
                )
                st.success("Done! Scroll down to take the quiz.")
            except Exception as e:
                st.error(f"Generation failed: {e}")

    st.divider()
    st.subheader("How this prototype works")
    st.markdown(
        """
- You choose **level + topic + difficulty**.
- The app makes **one LLM call** to generate a JSON question set.
- You answer questions.
- The app scores you and shows explanations.
"""
    )

with col2:
    st.subheader("Quiz")

    data = st.session_state.mcq_data
    if not data or "questions" not in data:
        st.info("Generate a question set to start.")
    else:
        questions: List[Dict[str, Any]] = data["questions"]

        # Render questions
        for q in questions:
            qid = str(q.get("id"))
            stem = q.get("stem", "").strip()
            choices = q.get("choices", {})
            st.markdown(f"**Q{qid}.** {stem}")

            options = ["A", "B", "C", "D"]
            labels = [f"{k}. {choices.get(k, '')}" for k in options]

            selected = st.radio(
                f"Select answer for Q{qid}",
                options=options,
                format_func=lambda k: f"{k}. {choices.get(k, '')}",
                key=f"radio_{qid}",
                index=None if not st.session_state.answers.get(qid) else options.index(st.session_state.answers[qid]),
            )

            if selected:
                st.session_state.answers[qid] = selected

            st.write("")

        # Submit
        if st.button("Submit answers"):
            st.session_state.submitted = True

        # Score + feedback
        if st.session_state.submitted:
            correct = 0
            total = len(questions)

            st.divider()
            st.subheader("Results")

            for q in questions:
                qid = str(q.get("id"))
                ans = (q.get("answer") or "").strip()
                user_ans = st.session_state.answers.get(qid)

                if user_ans == ans:
                    correct += 1
                    st.success(f"Q{qid}: Correct ({ans})")
                else:
                    st.error(f"Q{qid}: Incorrect. Correct answer: {ans}. Your answer: {user_ans}")

                exp = q.get("explanation", "").strip()
                if exp:
                    st.caption(exp)

            st.markdown(f"### Score: **{correct}/{total}**")

            # Lightweight guidance
            st.markdown("**Next steps (prototype):**")
            st.markdown("- Regenerate with a weaker topic area")
            st.markdown("- Increase difficulty once you're consistently >80%")
