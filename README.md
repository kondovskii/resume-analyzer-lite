# 📄 AI Resume Analyzer – Lite

An AI-powered web app that compares your resume to any job description and gives:
- An **overall fit score**
- Matched vs. missing skills
- A suggested improved resume bullet

### 🧠 Built With
- **Python 3.11**
- **Streamlit** (frontend)
- **OpenAI API** (`gpt-4o-mini` for analysis)
- **Playwright + BeautifulSoup** (to fetch job descriptions from links)
- **Dotenv** (for API key management)

---

### ⚙️ Features
✅ Upload or paste your resume  
✅ Paste or link any job posting  
✅ Automatic text extraction (PDF/DOCX/URL)  
✅ AI-powered fit score (semantic + LLM)  
✅ Highlights strong & weak areas  
✅ Suggests improved bullet points  

---

### 🏃 How to Run Locally

```bash
git clone https://github.com/kondovskii/resume-analyzer-lite.git
cd resume-analyzer-lite
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
