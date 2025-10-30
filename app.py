import os
import re
import requests
from bs4 import BeautifulSoup

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from parsers import read_pdf, read_docx
from scorer import embed, cosine

# -------------------- setup --------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Resume Analyzer — Lite", page_icon="📄", layout="wide")

# -------------------- minimal, safe styling (no custom wrappers) --------------------
st.markdown("""
<style>
:root{
  --bg:#f6f7fb; --text:#0f172a; --muted:#5f6b85;
  --brand:#2563eb; --brand-2:#1d4ed8;
}
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
.block-container { padding-top: 1.2rem; max-width: 1200px; }
h1,h2,h3,h4,h5,h6,label,p,div,span {
  color: var(--text); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}
small, .helptext { color: var(--muted) !important; }

/* Make native bordered containers a bit softer */
[data-testid="stContainer"] > div:has(> [data-testid="stVerticalBlock"]) { border-radius: 14px; }

/* Primary button */
div.stButton > button:first-child{
  background: linear-gradient(180deg, var(--brand), var(--brand-2));
  color:#fff; padding:.9rem 2.2rem; border-radius:12px; font-weight:700;
  border:1px solid #1e40af33; box-shadow:0 10px 22px rgba(37,99,235,.25);
}
div.stButton > button:hover{ filter:brightness(1.05) }

/* Progress bar */
.stProgress > div > div{ background: linear-gradient(90deg, var(--brand), var(--brand-2)) !important; }

/* Equal-height panels */
.equal-panel {
  min-height: 520px;          /* tune this if you want taller/shorter cards */
  display: flex;
  flex-direction: column;
  gap: 12px;
}
            

/* Tabs underline color (looks nicer) */
[data-baseweb="tab"] { font-weight:600; }
[data-baseweb="tab-border"]{ background: #e6e8ef !important; }
[data-baseweb="tab-highlight"]{ background: var(--brand-2) !important; }

  /* Centered action row for the button */
.action-row { display:flex; justify-content:center; }
.action-row > div { width: 520px; } /* button max width container */
          


/* IMPORTANT: neutralize BaseWeb focus bars INSIDE Streamlit widgets only */
section [data-baseweb]::before, section [data-baseweb]::after { content:none !important; display:none !important; }
</style>
""", unsafe_allow_html=True)

# -------------------- header --------------------
st.markdown("""
<div style="
  background: white;
  border-radius: 16px;
  padding: 28px 24px 22px 24px;
  box-shadow: 0 8px 18px rgba(15,23,42,0.06);
  margin-bottom: 28px;
  position: relative;
  z-index: 2;
">
  <h1 style="margin:0; display:flex; align-items:center; gap:10px; font-size:1.9rem;">📄 AI Resume Analyzer — Lite</h1>
  <p style="margin:.4rem 0 0; color:#5f6b85; font-size:1rem;">
    Provide a resume and a job description. We’ll compute a combined fit score and generate targeted improvements.
  </p>
</div>
""", unsafe_allow_html=True)

st.caption("v1.3 – balanced layout")


# -------------------- helpers --------------------
def extract_first_int_0_100(text: str):
    m = re.search(r'\b(100|[0-9]{1,2})\b', text)
    return int(m.group(1)) if m else None

def show_fetch_error(url: str, tried_js: bool):
    st.error("No job description text detected from that link.")
    st.markdown(
        "- Some job sites (e.g., **Workday / Greenhouse / Lever**) load content with JavaScript or block scraping.\n"
        f"- URL tried: `{url}`\n"
        f"- **Try this:** "
        f"{'You already tried JS rendering; some sites still block it. ' if tried_js else 'Turn on **Try JS rendering** and re-run, or '}copy–paste the job description into the text box on the right."
    )

@st.cache_data(show_spinner=False, ttl=300)
def fetch_url_text_simple(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Language": "en-CA,en-US;q=0.9,en;q=0.8"
        }
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript","iframe"]): tag.decompose()
        candidates = []
        for sel in ["main","article","[role=main]","#jobDescriptionText",".jobDescription",
                    ".jobs-description__container",".description",".content",".posting",".jobsearch-JobComponent"]:
            for el in soup.select(sel):
                txt = el.get_text(separator="\n", strip=True)
                if txt and len(txt) > 400: candidates.append(txt)
        if not candidates:
            txt = soup.get_text(separator="\n", strip=True)
            return txt if len(txt) > 200 else ""
        candidates.sort(key=len, reverse=True)
        return candidates[0][:20000]
    except Exception:
        return ""

@st.cache_data(show_spinner=False, ttl=300)
def fetch_url_text(url: str) -> str:
    text = fetch_url_text_simple(url)
    if len(text) >= 400:
        return text

    try_js = any(d in url.lower() for d in [
        "myworkdayjobs.com","workday","wd5.myworkdayjobs.com","greenhouse.io","boards.greenhouse.io","lever.co"
    ])
    if not try_js:
        return text

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
                viewport={"width":1280,"height":900}, locale="en-US", timezone_id="America/Toronto"
            )
            page = ctx.new_page()
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            selectors = ["article","main","[role=main]","#jobDescriptionText",".jobDescription",
                         ".jobs-description__container",".css-1p0xpbo",".css-1m3kac1",".description",".content"]
            for sel in selectors:
                try:
                    page.wait_for_selector(sel, timeout=6000); break
                except: continue
            page.wait_for_timeout(1500)
            html = page.content()
            browser.close()

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script","style","noscript","iframe"]): tag.decompose()
        candidates = []
        for sel in selectors + ["article","main"]:
            for el in soup.select(sel):
                txt = el.get_text(separator="\n", strip=True)
                if txt and len(txt) > 400: candidates.append(txt)
        if not candidates:
            txt = soup.get_text(separator="\n", strip=True)
            return txt if len(txt) > 200 else (text or "")
        candidates.sort(key=len, reverse=True)
        return candidates[0][:20000]
    except Exception:
        return text or ""

# -------------------- state --------------------
if "resume_text" not in st.session_state: st.session_state.resume_text = ""
if "jd_text" not in st.session_state: st.session_state.jd_text = ""
if "jd_url" not in st.session_state: st.session_state.jd_url = ""
if "try_js" not in st.session_state: st.session_state.try_js = True

# -------------------- layout (simple, balanced, centered button) --------------------
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("Resume")
        r_tabs = st.tabs(["Upload", "Paste Text"])
        with r_tabs[0]:
            uploaded = st.file_uploader("Upload Resume", type=["pdf", "docx"], key="resume_uploader")
            if uploaded is not None:
                data = uploaded.read()
                txt = read_pdf(data) if uploaded.name.lower().endswith(".pdf") else read_docx(data)
                if txt.strip():
                    st.session_state.resume_text = txt
                    st.success(f"Loaded text from {uploaded.name}")
                else:
                    st.warning("Could not extract text. Try DOCX or switch to Paste Text.")
        with r_tabs[1]:
            st.session_state.resume_text = st.text_area(
                "Paste Resume Text",
                value=st.session_state.resume_text,
                height=280,   # match the JD field height
                placeholder="Paste your resume here…"
            )

with right:
    with st.container(border=True):
        st.subheader("Job Description")
        j_tabs = st.tabs(["Paste Text", "From URL"])
        with j_tabs[0]:
            st.session_state.jd_text = st.text_area(
                "Paste Job Description",
                value=st.session_state.jd_text,
                height=280,   # matches resume text area
                placeholder="Paste the job description here…"
            )
            jd_method = "Paste Text"
            st.session_state.jd_url = ""
        with j_tabs[1]:
            st.session_state.jd_url = st.text_input(
                "Job posting URL",
                value=st.session_state.jd_url,
                placeholder="https://…"
            )
            st.session_state.try_js = st.checkbox(
                "Try JS rendering if needed (slower)",
                value=st.session_state.try_js
            )
            st.session_state.jd_text = st.text_area(
                "JD Preview (editable)",
                value=st.session_state.jd_text,
                height=220
            )
            jd_method = "From URL"

# -- centered action row (looks consistent regardless of left/right heights) --
st.markdown('<div class="action-row">', unsafe_allow_html=True)
col = st.columns([1])[0]
with col:
    analyze = st.button("🔍 Analyze Resume Fit", type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)



# -------------------- analysis --------------------
if analyze:
    # Fetch JD if URL mode
    jd_text = st.session_state.jd_text
    tried_js = False
    if jd_method == "From URL":
        url_clean = st.session_state.jd_url.strip()
        if url_clean:
            if st.session_state.try_js:
                jd_text = fetch_url_text(url_clean); tried_js = True
            else:
                jd_text = fetch_url_text_simple(url_clean); tried_js = False
            if not jd_text or len(jd_text.strip()) < 200:
                show_fetch_error(url_clean, tried_js)
                jd_text = st.session_state.jd_text
        else:
            st.error("No URL provided — please enter a link or paste the job description.")
            jd_text = st.session_state.jd_text

    resume_text = st.session_state.resume_text
    if not resume_text.strip() or not jd_text.strip():
        st.error("Please provide BOTH: resume (left) and job description (right).")
        st.stop()

    # Trim long inputs (speed/cost)
    resume_text = resume_text[:15000]
    jd_text = jd_text[:15000]

    # 1) Semantic score
    sem_score = None
    with st.spinner("Computing semantic similarity…"):
        try:
            r_vec = embed(resume_text)
            j_vec = embed(jd_text)
            sem = cosine(r_vec, j_vec)
            sem_score = int(round(sem * 100))
        except Exception as e:
            st.warning(f"Could not compute embeddings: {e}")

    # 2) LLM analysis with numeric score first
    prompt = f"""
You are a precise resume reviewer.
First, output ONLY a single integer from 0 to 100 indicating the Fit Score (no words before or after).
Then on new lines, provide:
- Top 5 matched skills/experiences (bullets)
- Top 5 missing/weak areas (bullets)
- One improved resume bullet (≤30 words, include 1 metric, natural phrasing)

Resume:
{resume_text}

Job Description:
{jd_text}
"""
    ai_text, ai_score = "", None
    with st.spinner("Analyzing with AI…"):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[{"role":"user","content":prompt}]
            )
            ai_text = resp.choices[0].message.content or ""
            ai_score = extract_first_int_0_100(ai_text)
        except Exception as e:
            st.error(f"Error contacting OpenAI API: {e}")

    # 3) Combined score
    if sem_score is not None and ai_score is not None:
        final_score = int(round(0.6*sem_score + 0.4*ai_score))
        st.subheader(f"Overall Fit Score: {final_score}/100")
        st.progress(min(max(final_score,0),100)/100)
        with st.expander("Details"):
            st.write(f"- Semantic score: **{sem_score}/100**")
            st.write(f"- AI score: **{ai_score}/100**")
    elif sem_score is not None:
        st.subheader(f"Fit Score: {sem_score}/100")
        st.progress(min(max(sem_score,0),100)/100)
        st.info("AI score unavailable; showing semantic score only.")
    elif ai_score is not None:
        st.subheader(f"Fit Score: {ai_score}/100")
        st.progress(min(max(ai_score,0),100)/100)
        st.info("Semantic score unavailable; showing AI score only.")
    else:
        st.info("Couldn’t compute a numeric score. See analysis below.")

    # 4) Show AI details (strip the leading number line)
    if ai_text.strip():
        lines = ai_text.strip().splitlines()
        if lines and extract_first_int_0_100(lines[0]) is not None:
            ai_text_clean = "\n".join(lines[1:]).strip()
        else:
            ai_text_clean = ai_text
        st.markdown(ai_text_clean if ai_text_clean else ai_text)
