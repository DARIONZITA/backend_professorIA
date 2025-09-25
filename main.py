from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import tempfile
import requests
import time
import uuid
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
import re
from copy import deepcopy

# Módulos dinâmicos
from analysis_engine import analyze_text
from grouping_engine import build_groups, build_class_insights

# MOTOR NOVO - IMPORTAÇÃO DO MOTOR ELÉTRICO DE ALTA VELOCIDADE
from transcription_engine import transcrever_imagem_com_gemini

# Database helpers
from db import (
    list_students as db_list_students,
    get_student_by_id as db_get_student_by_id,
    create_student as db_create_student,
    delete_student as db_delete_student,
    list_analyses as db_list_analyses,
    get_analysis_by_id as db_get_analysis_by_id,
    insert_analysis as db_insert_analysis,
    list_analyses_by_student as db_list_analyses_by_student,
    list_analyses_grouped_by_class as db_list_analyses_grouped_by_class,
)


app = FastAPI(title="Professor AI - Radar Pedagógico", version="1.0.0")

# Ensure public directory for images and mount StaticFiles
public_temp_dir = Path("temp")
public_images_dir = public_temp_dir / "images"
public_images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/temp", StaticFiles(directory=str(public_temp_dir)), name="temp")

# Configure CORS to allow frontend
app.add_middleware(
    CORSMiddleware,
    # During development allow all origins to avoid CORS issues from different hostnames
    # In production restrict this to the known frontend origin(s)
    allow_origins=["https://v0-ai-tutor-app-ten.vercel.app"],  # Frontend Next.js
    # Allow vercel subdomains and localhost for debugging preflight requests
    allow_origin_regex=r"https?://(.+\.vercel\.app|localhost(:\d+)?)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Space configurations
SPACE_URL = 'https://dnzita-professorIa.hf.space'
LANGUAGES = ['English', 'Swedish', 'Norwegian', 'Medieval', 'Math']
DEFAULT_LANGUAGE = 'English'
POLL_INTERVAL = 5
TIMEOUT = 15 * 60

# Pydantic classes for validation
class Student(BaseModel):
    id: str
    name: str
    class_name: str

class AnalysisData(BaseModel):
    imageUrl: str
    detected_text: str
    mainError: str
    errorPercentage: int
    concepts: List[str]
    suggestions: List[str]
    reasoning: Optional[str] = None

class Analysis(BaseModel):
    id: str
    studentName: str
    subject: str
    timestamp: datetime
    data: AnalysisData

class CreateStudentRequest(BaseModel):
    name: str
    class_name: str

def _build_absolute_image_url(image_url: Optional[str], request: Request) -> Optional[str]:
    if not image_url:
        return image_url
    if image_url.startswith("http://") or image_url.startswith("https://"):
        return image_url
    base = str(request.base_url).rstrip("/")
    normalized = image_url if image_url.startswith("/") else f"/{image_url}"
    return f"{base}{normalized}"


def _prepare_analysis_for_response(analysis: Dict[str, Any], request: Request) -> Dict[str, Any]:
    prepared = deepcopy(analysis)
    data = prepared.get("data") or {}
    if isinstance(data, dict):
        image_url = data.get("imageUrl") or data.get("image_url")
        if image_url:
            data = dict(data)
            data.pop("image_url", None)
            data["imageUrl"] = _build_absolute_image_url(image_url, request)
            prepared["data"] = data
    return prepared


# Intelligent analysis system based on OCR text
def analyze_pedagogical(detected_text: str, subject: str) -> Dict[str, Any]:
    """Wrapper for dynamic analysis via LLM/heuristic."""
    return analyze_text(detected_text, subject)


def compute_historical_summary(student_name: str, subject: Optional[str] = None) -> Optional[str]:
    """Aggregates previous analyses from the same student and returns a textual summary.

    The function collects only already computed fields (mainError, concepts, suggestions,
    ai_structured.specificError, ai_structured.isRecurrent) to avoid reprocessing
    texts and spending LLM tokens. Returns None if there is no history.
    """
    if not student_name:
        return None
    relevant = db_list_analyses_by_student(student_name, subject)
    if not relevant:
        return None

    # Simple counters
    from collections import Counter
    main_errors = Counter()
    concepts = Counter()
    suggestions = Counter()
    recurring_count = 0
    total = 0

    for a in relevant:
        total += 1
        data = a.get('data', {})
        # Prefer fields from ai_structured when available (more specific)
        ai_struct = data.get('ai_structured') or {}
        if ai_struct.get('specificError'):
            main_errors[ai_struct.get('specificError')] += 1
        elif data.get('mainError'):
            main_errors[data.get('mainError')] += 1

        # concepts: can be in ai_analysis or top-level
        ai_analysis = data.get('ai_analysis') or {}
        for c in ai_analysis.get('concepts', []) or data.get('concepts', []) or []:
            try:
                concepts[str(c)] += 1
            except Exception:
                continue

        for s in ai_analysis.get('suggestions', []) or data.get('suggestions', []) or []:
            try:
                suggestions[str(s)] += 1
            except Exception:
                continue

        if ai_struct.get('isRecurrent'):
            recurring_count += 1

    # Select top-n
    top_errors = [k for k, _ in main_errors.most_common(3)]
    top_concepts = [k for k, _ in concepts.most_common(5)]
    top_suggestions = [k for k, _ in suggestions.most_common(5)]

    # Build summarized text without using LLM
    parts = []
    parts.append(f"Historical summary based on {total} previous analyses for {student_name} ({subject}).")
    if top_errors:
        parts.append(f"Most frequent issues: {', '.join(top_errors)}.")
    if top_concepts:
        parts.append(f"Related concepts often involved: {', '.join(top_concepts)}.")
    if recurring_count:
        parts.append(f"Detected {recurring_count} cases flagged as recurrent patterns.")
    if top_suggestions:
        parts.append(f"Common suggestions previously given: {', '.join(top_suggestions)}.")

    # Short diagnostic sentence
    parts.append("Recommendation: focus targeted practice on the most frequent issues and review the related concepts listed above.")

    return " ".join(parts)

@app.get('/')
def read_root():
    return {'Status': 'The Pedagogical Radar engine is running!', 'version': '2.0.0'}


@app.head('/')
def head_root():
    """Lightweight HEAD endpoint for health/status checks returning 200 without body."""
    return Response(status_code=200)

def upload_image_to_space(image_path: str) -> Optional[str]:
    '''Upload image to Space and return server path.'''
    with open(image_path, 'rb') as f:
        files = {'files': f}
        r = requests.post(f'{SPACE_URL}/upload', files=files, timeout=60)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, str):
            return first
        elif isinstance(first, dict) and 'path' in first:
            return first['path']
    elif isinstance(data, dict) and 'path' in data:
        return data['path']
    return None

def start_job_from_server_path(path: str, language: str) -> Optional[str]:
    '''Start job on Space.'''
    payload = {'path': path, 'language': language}
    r = requests.post(f'{SPACE_URL}/jobs/start_from_path', json=payload, timeout=30)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    return data.get('job_id')

def get_job_status(job_id: str) -> Optional[dict]:
    '''Query job status.'''
    r = requests.get(f'{SPACE_URL}/jobs/{job_id}', timeout=30)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None

def poll_job_until_complete(job_id: str) -> Optional[dict]:
    '''Wait for job completion.'''
    start = time.time()
    while True:
        if time.time() - start > TIMEOUT:
            return None
        status = get_job_status(job_id)
        if not status:
            time.sleep(POLL_INTERVAL)
            continue
        if status.get('status') in ('completed', 'failed', 'cancelled'):
            return status
        time.sleep(POLL_INTERVAL)

# ==================== ENDPOINTS DE ESTUDANTES ====================

@app.get('/students')
def get_students(class_name: Optional[str] = None):
    """Returns list of students. Optional query param `class_name` filters results."""
    students = db_list_students(class_name=class_name)
    return {'students': students}

@app.post('/students')
def create_student(student_data: CreateStudentRequest):
    """Creates a new student. Returns 201 on success, 409 on duplicate."""
    new_student = {
        "id": str(uuid.uuid4()),
        "name": student_data.name,
        "class_name": student_data.class_name
    }
    try:
        created = db_create_student(new_student)
        # Return the created student in a predictable shape and 201 status
        return JSONResponse(status_code=201, content={"student": created})
    except ValueError as ve:
        # Duplicate student
        raise HTTPException(status_code=409, detail=str(ve))


@app.delete('/students/{student_id}')
def remove_student(student_id: str):
    """Deletes a student by id. Returns 204 on success, 404 if not found."""
    ok = db_delete_student(student_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Student not found")
    return {}, 204

@app.get('/students/{student_id}')
def get_student(student_id: str):
    """Returns a specific student"""
    student = db_get_student_by_id(student_id)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


# -------------------- Class endpoints --------------------
@app.get('/classes')
def get_classes():
    """Return list of classes with student counts."""
    students = db_list_students()
    classes: Dict[str, int] = {}
    for s in students:
        cls = s.get('class_name') or 'Unknown'
        classes[cls] = classes.get(cls, 0) + 1
    items = [{"class_name": k, "student_count": v} for k, v in sorted(classes.items())]
    return {"classes": items}


@app.get('/classes/{class_name}')
def get_class_insights(class_name: str, request: Request, force: Optional[bool] = False):
    """Return class-level insights (LLM-powered). Query param force=true to recompute."""
    # Get students in class
    students = db_list_students(class_name)
    if not students:
        raise HTTPException(status_code=404, detail="Class not found or has no students")

    # Collect most recent analysis per student
    all_analyses = []
    for s in students:
        name = s.get('name')
        analyses = db_list_analyses_by_student(name)
        if analyses:
            # keep the most recent
            all_analyses.append(analyses[0])

    # If no analyses, return informative empty structure
    insights = build_class_insights(all_analyses, class_name, force=bool(force))
    # Ensure image URLs in detailed items are absolute if present in analysis objects
    # Prepare detailed list if it contains analysis objects
    if isinstance(insights.get('detailed'), list):
        # Nothing to transform here because detailed items are simple dicts; keep as-is.
        pass

    return insights

# ==================== ENDPOINTS DE ANÁLISES ====================

@app.get('/analyses')
def get_analyses(request: Request):
    """Returns all analyses"""
    analyses = db_list_analyses()
    prepared = [_prepare_analysis_for_response(a, request) for a in analyses]
    return {'analyses': prepared}

@app.get('/analyses/{analysis_id}')
def get_analysis(analysis_id: str, request: Request):
    """Returns a specific analysis"""
    analysis = db_get_analysis_by_id(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return _prepare_analysis_for_response(analysis, request)

@app.post('/analyze_exercise')
async def analyze_exercise(
    request: Request,
    file: UploadFile = File(...), 
    student_id: Optional[str] = Form(None),
    subject: str = Form("Mathematics")
):
    '''Analyzes complete exercise: OCR + Pedagogical Analysis'''

    # Check if student exists
    student_name = "Unknown Student"
    if student_id:
        student = db_get_student_by_id(student_id)
        if student:
            student_name = student['name']

    # Salvar arquivo temporariamente
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / f'temp_{file.filename}'
    try:
        with open(temp_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        # === SUBSTITUIÇÃO CIRÚRGICA: MOTOR ANTIGO → MOTOR NOVO ===
        
        # MOTOR ANTIGO (TrOCR - COMENTADO):
        # server_path = upload_image_to_space(str(temp_path))
        # if not server_path:
        #     return {'error': 'Failed to upload image'}
        # job_id = start_job_from_server_path(server_path, language)
        # if not job_id:
        #     return {'error': 'Failed to start OCR processing'}
        # final_status = poll_job_until_complete(job_id)
        # if not final_status:
        #     return {'error': 'Timeout waiting for OCR result'}
        # if final_status.get('status') != 'completed':
        #     return {'error': 'OCR processing failed', 'status': final_status.get('status'), 'details': final_status.get('error')}
        # detected_text = final_status.get('result', '')
        
        # MOTOR NOVO (GEMINI 1.5 FLASH - ATIVO):
        # Ler os bytes da imagem
        with open(temp_path, 'rb') as f:
            image_bytes = f.read()
        
        # Transcrever com o motor elétrico de alta velocidade
        detected_text = transcrever_imagem_com_gemini(image_bytes)
        
        if not detected_text:
            return {'error': 'Failed to transcribe image with Gemini 1.5 Flash'}

        # 4. Intelligent pedagogical analysis
        pedagogical_analysis = analyze_pedagogical(detected_text, subject)

        # 5. Create analysis entry
        analysis_id = str(uuid.uuid4())

        # Save a public copy of the image (for later display)
        suffix = Path(file.filename).suffix or '.jpg'
        stored_image_path = public_images_dir / f"{analysis_id}{suffix}"
        try:
            # copy from temporary file that has already been written
            with open(temp_path, 'rb') as src, open(stored_image_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        except Exception:
            # not fatal, just log
            pass

        image_relative_path = f"/temp/images/{stored_image_path.name}"

        # Ensure ai_structured exists and fill historicalAnalysis from previous analyses to avoid extra LLM calls
        ai_struct = pedagogical_analysis.get("ai_structured") or {}
        if not ai_struct.get('historicalAnalysis'):
            hist = compute_historical_summary(student_name, subject)
            if hist:
                ai_struct['historicalAnalysis'] = hist

        # If the analysis provided a studentFeedback template, fill the student's name
        student_feedback = None
        try:
            sf = pedagogical_analysis.get('studentFeedback')
            if isinstance(sf, str):
                student_feedback = sf.replace('{student_name}', student_name)
        except Exception:
            student_feedback = None

        analysis_data = {
            "imageUrl": image_relative_path,
            "detected_text": detected_text,
            "mainError": pedagogical_analysis.get("mainError"),
            "errorPercentage": pedagogical_analysis.get("errorPercentage"),
            "concepts": pedagogical_analysis.get("concepts", []),
            "suggestions": pedagogical_analysis.get("suggestions", []),
            "reasoning": pedagogical_analysis.get("reasoning"),
            "ai_analysis": pedagogical_analysis.get("ai_analysis"),
            "ai_structured": ai_struct
            ,
            "score": pedagogical_analysis.get("score"),
            "studentFeedback": student_feedback
        }
        
        new_analysis = {
            "id": analysis_id,
            "studentName": student_name,
            "subject": subject,
            "timestamp": datetime.now().isoformat(),
            "data": analysis_data
        }
        
        # 6. Save analysis
        db_insert_analysis(new_analysis)

        response_analysis = _prepare_analysis_for_response(new_analysis, request)

        return {
            'analysis': response_analysis,
            'ocr_info': {
                'filename': file.filename,
                'processing_time': '< 3 segundos (motor Gemini)',  # Motor novo é muito mais rápido
                'transcription_engine': 'Gemini 1.5 Flash (novo motor elétrico)'
            }
        }

    finally:
        # Clean up temporary file
        try:
            temp_path.unlink(missing_ok=True)
            Path(temp_dir).rmdir()
        except:
            pass

# ==================== GROUPING ENDPOINT ====================

@app.get('/student_groups')
def get_student_groups(force: bool = False):
    """Dynamic grouping via LLM (or heuristic). Parameter force=true recalculates."""
    analyses = db_list_analyses()
    grouping = build_groups(analyses, force=force)
    return grouping

@app.post('/student_groups/recompute')
def recompute_groups():
    analyses = db_list_analyses()
    grouping = build_groups(analyses, force=True)
    return grouping


@app.get('/analyses_by_class')
def get_analyses_by_class(request: Request):
    """Return analyses grouped by class (class_name).

    Each group contains 'class_name', 'analyses', 'count' and 'average_error'.
    """
    groups = db_list_analyses_grouped_by_class()
    # ensure image URLs are absolute
    prepared = []
    for g in groups:
        grp = dict(g)
        grp_analyses = grp.get('analyses') or []
        prepared_analyses = [_prepare_analysis_for_response(a, request) for a in grp_analyses]
        grp['analyses'] = prepared_analyses
        prepared.append(grp)
    return {'groups': prepared}

# ==================== SIMPLIFIED ENDPOINT FOR COMPATIBILITY ====================

@app.post('/analyze_simple')
async def analyze_simple(file: UploadFile = File(...)):
    '''Simplified version for compatibility - OCR only'''

    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / f'temp_{file.filename}'
    try:
        with open(temp_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        # === SUBSTITUIÇÃO CIRÚRGICA: MOTOR ANTIGO → MOTOR NOVO ===
        
        # MOTOR ANTIGO (TrOCR - COMENTADO):
        # server_path = upload_image_to_space(str(temp_path))
        # if not server_path:
        #     return {'error': 'Failed to upload image'}
        # job_id = start_job_from_server_path(server_path, language)
        # if not job_id:
        #     return {'error': 'Failed to start processing'}
        # final_status = poll_job_until_complete(job_id)
        # if not final_status:
        #     return {'error': 'Timeout waiting for result'}
        # if final_status.get('status') == 'completed':
        
        # MOTOR NOVO (GEMINI 1.5 FLASH - ATIVO):
        # Ler os bytes da imagem
        with open(temp_path, 'rb') as f:
            image_bytes = f.read()
        
        # Transcrever com o motor elétrico de alta velocidade
        detected_text = transcrever_imagem_com_gemini(image_bytes)
        
        if detected_text:
            return {
                'filename': file.filename,
                'detected_text': detected_text,  # Resultado do motor Gemini
                'processing_time': '< 3 segundos (motor Gemini)',
                'transcription_engine': 'Gemini 1.5 Flash (novo motor elétrico)'
            }
        else:
            return {
                'error': 'Failed to transcribe image with Gemini 1.5 Flash',
                'transcription_engine': 'Gemini 1.5 Flash (novo motor elétrico)'
            }

    finally:
        # Clean up temporary file
        try:
            temp_path.unlink(missing_ok=True)
            Path(temp_dir).rmdir()
        except:
            pass
