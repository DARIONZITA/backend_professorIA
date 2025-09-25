Professor AI - Pedagogical Radar
================================

Overview
--------
This repository contains a small backend for an educational assistance tool called "Professor AI - Pedagogical Radar". The service provides endpoints to upload student exercise images, run OCR (either a legacy remote OCR service or an integrated Gemini multimodal model), analyze the recognized text with an LLM to extract pedagogical insights, and group students into learning groups.

Main components
---------------
- `main.py` - FastAPI application that exposes HTTP endpoints for students, analyses and class/group insights. Handles file uploads and orchestrates OCR + analysis.
- `transcription_engine.py` - Implements two OCR paths:
  - TrOCR (legacy) via an external Space with an upload/start/poll flow (see `job_client.py`).
  - Gemini multimodal (fast) path using `google.generativeai` if `GEMINI_API_KEY` is provided.
- `analysis_engine.py` - Transforms OCR text into structured pedagogical JSON using the configured LLM or a heuristic fallback if the LLM is unavailable.
- `grouping_engine.py` - Groups analyses into pedagogical learning groups using the LLM or a heuristic fallback.
- `llm_client.py` - Thin adapter for text LLM providers. Prefers Groq (`API_GROQ`) and falls back to Gemini (`GEMINI_API_KEY`). Attempts to extract JSON from model outputs.
- `db.py` - MongoDB helpers (init, students and analyses collections, seeding defaults, import from JSON if present).
- `job_client.py` - Helper for uploading images and starting/polling asynchronous OCR jobs on a remote Space.

Environment variables (important)
---------------------------------
Create a `.env` file in the project root or export these variables in your environment. **Do not commit `.env` to version control.** Use `.env.example` as a template.

Required for basic operation:
- `MONGO_URI` - MongoDB connection string (example: mongodb+srv://user:pass@cluster...)
- `MONGO_DB_NAME` - (optional) database name, default `professorai`

Optional (for LLM/OCR features):
- `API_GROQ` - API key for Groq (preferred LLM provider in this app).
- `API_GROQ_URL` - custom Groq endpoint (optional).
- `GROQ_MODEL` - model name for Groq (optional).
- `GEMINI_API_KEY` - Google Gemini API key (used for multimodal OCR and fallback LLM).
- `GEMINI_MODEL` - Gemini model name (defaults to `gemini-1.5-flash-latest`).
- `SPACE_URL` - URL of the remote OCR Space if you use the legacy TrOCR flow. Defaults to the value in code.
- `GROUPING_CACHE_TTL` - cache ttl in seconds for grouping results (defaults to 120).

Installation
------------
Recommended to use a virtual environment.

Windows (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your actual values:

```powershell
cp .env.example .env
# Edit .env with your real API keys and MongoDB URI
```

**Important:** The repository includes a `.gitignore` file that excludes `.env` from version control. Never commit your actual secrets.

If you want to use the Gemini multimodal OCR or the Gemini LLM paths, install the `google-generativeai` package and set `GEMINI_API_KEY` in your environment.

Run (development)
-----------------
Start the FastAPI app (development reload enabled):

```powershell
cd "C:\Users\DELL\OneDrive\Documents\hacktohns\professor assistence"
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Notes:
- The code mounts the `temp/` directory as static at `/temp` to serve uploaded images.
- If MongoDB is not available, `db.init_db()` will raise an error. You may run a local MongoDB or set `MONGO_URI` to a working Atlas cluster.

API highlights
--------------
- GET `/` - health/status
- GET `/students` - list students
- POST `/students` - create student (JSON body: {"name": "Name", "class_name": "5th A"})
- POST `/analyze_exercise` - multipart/form-data with `file` (image) and optional `student_id` and `subject` form fields. The endpoint returns the OCR detection and pedagogical analysis.
- GET `/analyses` - list analyses
- GET `/classes/{class_name}` - class-level insights (LLM-powered when keys available)
- POST `/student_groups/recompute` - (if present) recompute grouping (depends on implementation)

Security & secrets
------------------
- Do not commit `.env` to source control. The repository currently contains an example `.env` (from attachments) which should be removed or rotated because it may contain real keys.
- Use `.env.example` as a template for your local setup.
- If you accidentally committed secrets, rotate the keys immediately and consider the old ones compromised.

Code review notes and recommendations
------------------------------------
Strengths:
- Clear modular structure separating OCR, analysis, grouping, LLM client and DB layers.
- Thoughtful fallbacks when LLM or external services are unavailable.
- Cache and seeding mechanisms in `db.py` and `grouping_engine.py`.

Issues / risks to address:
1. Secrets in `.env` included in the attachment (GEMINI_API_KEY, API_GROQ, MONGO_URI). Rotate these keys immediately and remove `.env` from the repository. Treat as compromised.
2. Error handling: some functions raise RuntimeError on missing env (e.g., `db.init_db()`); consider returning usable messages on startup.
3. Tests: there are no automated tests. Add at least a couple of unit tests for `analysis_engine.fallback_analysis` and `grouping_engine.fallback_groups`.
4. Requirements: added `pillow` and `python-multipart` because `transcription_engine.py` uses PIL and FastAPI file upload endpoints need `python-multipart`.
5. Timeouts and polling: `poll_job_until_complete` uses a global TIMEOUT; consider making it configurable and logging progress.
6. JSON extraction from LLM: brittle. Consider using structured output features from your provider or asking for strict JSON with synthetic delimiters.

Suggested small improvements to implement next:
- Add basic unit tests for heuristic functions.
- Add health / readiness endpoint that checks MongoDB and LLM availability.
- Add graceful startup logs describing which features (Gemini/Groq/Mongo) are enabled.

License
-------
No license specified. Add an appropriate license file if you plan to publish this project.

Contact
-------
If you want, I can also:
- Add unit tests and a GitHub Actions workflow to run them.
- Create a small healthcheck endpoint and a startup log summary.
- Remove or rotate the `.env` file and replace it with an `.env.example` that does not contain secrets.

```text
Summary of changes made in this review:
- Updated `requirements.txt` to add `pillow` and `python-multipart`.
- Added this `README.md` with usage and recommendations.
```
