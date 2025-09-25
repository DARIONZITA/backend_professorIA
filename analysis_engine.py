"""Dynamic pedagogical analysis module using LLM.

Responsible for transforming OCR text into structured insights.
"""
from __future__ import annotations
from typing import Dict, Any
import math
import re
from llm_client import generate_structured, llm_available

ANALYSIS_SYSTEM = (
    "You are a pedagogical assistant. Generate strict JSON describing student difficulties. "
    "Do not invent non-existent content beyond the provided text and subject context."
)

ANALYSIS_SCHEMA_INSTRUCTIONS = (
    "Return ONLY a markdown block with valid JSON containing keys: "
    "mainError (string), errorPercentage (0-100 int), concepts (list of up to 8 strings), "
    "suggestions (list of up to 8 strings), reasoning (short string). Do not include explanations outside the JSON block."
)

def fallback_analysis(detected_text: str, subject: str) -> Dict[str, Any]:
    words = detected_text.split()
    length_factor = min(100, max(10, len(words)))
    # Simple heuristic to derive a pseudo "error percentage"
    error_percentage = min(85, max(15, int(math.log2(length_factor + 1) * 12)))
    return {
        "mainError": "Pending advanced analysis (LLM disabled)",
        "errorPercentage": error_percentage,
        "concepts": [subject[:40] or "General"],
        "suggestions": ["Enable GEMINI_API_KEY to obtain richer insights"],
        "reasoning": "Fallback heuristic",
        "ai_analysis": None
    }


def _detect_total_exercises(detected_text: str) -> int:
    """Tentativa heurística de contar quantos exercícios existem no texto.

    Procura por linhas numeradas (1., 2) ), palavras 'question/questão', pontos de interrogação
    ou parágrafos separados. Se não encontrar padrões, retorna 1 como fallback.
    """
    if not detected_text or not detected_text.strip():
        return 1
    # linhas numeradas: "1.", "2)" etc.
    lines = detected_text.splitlines()
    numbered = sum(1 for L in lines if re.match(r"^\s*\d+\s*[\.|\)]", L))
    if numbered >= 2:
        return numbered

    # busca por palavras 'question', 'questão' (pt/eng)
    qmatches = len(re.findall(r"\b(?:question|questão|questoes|questões|q)\b", detected_text, flags=re.I))
    if qmatches >= 2:
        return qmatches

    # procura por linhas terminadas em ')' tipo '1)'
    paren_nums = len(re.findall(r"\b\d+\)", detected_text))
    if paren_nums >= 2:
        return paren_nums

    # número de interrogações pode indicar perguntas
    qm = detected_text.count('?')
    if qm >= 2:
        return qm

    # fallback: parágrafos significativos
    paragraphs = [p for p in re.split(r"\n\s*\n", detected_text) if len(p.strip()) > 30]
    if len(paragraphs) >= 2:
        return len(paragraphs)

    return 1

def analyze_text(detected_text: str, subject: str) -> Dict[str, Any]:
    if not detected_text.strip():
        return {
            "mainError": "Empty or unreadable submission",
            "errorPercentage": 100,
            "concepts": [],
            "suggestions": ["Send a clearer image", "Check lighting and focus"],
            "reasoning": "No text recognized"
        }
    if not llm_available():
        return fallback_analysis(detected_text, subject)
    user_prompt = f"""
OCR Text (limited / sanitized):\n{detected_text[:4000]}\n---\nSubject/Context: {subject}\n\n{ANALYSIS_SCHEMA_INSTRUCTIONS}
"""
    result = generate_structured(user_prompt, system=ANALYSIS_SYSTEM, temperature=0.2)
    if not result.get("success"):
        return fallback_analysis(detected_text, subject)
    data = result["data"]
    # Minimal sanitization
    main_error = str(data.get("mainError", "Unspecified"))[:200]
    error_pct = data.get("errorPercentage", 50)
    try:
        error_pct = int(error_pct)
    except Exception:
        error_pct = 50
    error_pct = max(0, min(100, error_pct))
    concepts = [str(c)[:80] for c in data.get("concepts", [])][:8]
    suggestions = [str(s)[:120] for s in data.get("suggestions", [])][:8]
    reasoning = str(data.get("reasoning", ""))[:500]
    # Try to also produce the legacy pedagogical schema (mainConcept, specificError, ...)
    ai_structured = None
    try:
        # Prompt the LLM to return the legacy compact pedagogical JSON
        legacy_system = (
            "You are a specialized pedagogical assistant. Analyze the student's text and return ONLY a JSON with these specific keys: "
            "mainConcept (main concept being studied), "
            "specificError (specific error identified in the text), "
            "isRecurrent (boolean - if it's a common/recurrent error in this type of exercise), "
            "historicalAnalysis (detailed analysis of patterns and historical context of student errors), "
            "suggestionForTeacher (specific and practical suggestion for the teacher), "
            "generatedMicroExercise (list of 2-3 micro-exercises in object format with 'sentence' and 'answer'). "
            "Be detailed in historicalAnalysis and specific in suggestions. Do not add text outside the JSON."
        )
        legacy_prompt = f"""Student OCR text: {detected_text[:3000]}

Context/Subject: {subject}
Preliminary error summary: {main_error}

Analyze deeply:
1. What is the main concept being worked on?
2. What specific error was made?
3. Is this error common/recurrent in this type of exercise?
4. Provide a detailed historical analysis about the patterns of this type of error
5. Give a specific and practical suggestion for the teacher
6. Generate 2-3 micro-exercises targeted to correct this specific error"""
        legacy_res = generate_structured(legacy_prompt, system=legacy_system, temperature=0.2)
        if legacy_res.get("success") and isinstance(legacy_res.get("data"), dict):
            ai_structured = legacy_res.get("data")
    except Exception:
        ai_structured = None

    # Build a simple "score" (ex.: 5/6) using heuristics to detect how many
    # exercises exist in the submission and the error percentage produced by the LLM.
    total_ex = _detect_total_exercises(detected_text)
    correct = max(0, min(total_ex, int(round((100 - error_pct) / 100.0 * total_ex))))
    score = {
        "correct": correct,
        "total": total_ex,
        "label": f"{correct}/{total_ex}"
    }

    # Build a student-facing feedback template (in English) with a placeholder
    # for the student's name: {student_name}
    # Prefer to include the first generated micro-exercise sentence if available.
    micro_sent = None
    try:
        gen_ex = data.get('generatedMicroExercise') or ai_structured.get('generatedMicroExercise')
        if isinstance(gen_ex, list) and len(gen_ex) > 0:
            first = gen_ex[0]
            # Accept multiple possible keys for the exercise prompt
            micro_sent = (first.get('sentence') if isinstance(first, dict) else None) or first.get('prompt') if isinstance(first, dict) else None
            if not micro_sent and isinstance(first, str):
                micro_sent = first
    except Exception:
        micro_sent = None

    # Short guidance extracted from suggestions or reasoning
    guidance = None
    try:
        if suggestions:
            guidance = suggestions[0]
        elif reasoning:
            guidance = reasoning.split('\n')[0][:150]
    except Exception:
        guidance = None

    if micro_sent:
        student_feedback_template = (
            "Hi {student_name}! I reviewed your work and noticed you struggled with: \"%s\". "
            "Here's a short tip: %s. Try this short practice: %s"
        ) % (main_error, guidance or "follow the steps above", micro_sent)
    else:
        student_feedback_template = (
            "Hi {student_name}! I reviewed your work and noticed you struggled with: \"%s\". "
            "Here's a short tip: %s. Keep practicing and try similar exercises to improve."
        ) % (main_error, guidance or "review the related concept")

    # Return both normalized fields and the raw structured JSON from the LLM plus legacy structure
    return {
        "mainError": main_error,
        "errorPercentage": error_pct,
        "concepts": concepts,
        "suggestions": suggestions,
        "reasoning": reasoning,
        "ai_analysis": data,
        "ai_structured": ai_structured,
        "score": score,
        "studentFeedback": student_feedback_template,
    }
