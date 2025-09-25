"""Dynamic grouping engine using LLM.

Receives list of analyses and generates explainable clusters.
"""
from __future__ import annotations
from typing import List, Dict, Any
import time, hashlib
from llm_client import generate_structured, llm_available

GROUP_SYSTEM = (
    "You are a pedagogical expert. Your job is to analyze short per-student analysis records and produce a compact clustering of students into learning groups. "
    "STRICT REQUIREMENTS: Output only valid JSON (no explanatory text, no markdown). The top-level JSON must be an object with a single key `groups` whose value is a list." 
    "Each group must be an object with the following keys: \n"
    "- id: short slug string (use lowercase, hyphen-separated)\n"
    "- name: human-friendly name\n"
    "- level: one of \"high\", \"medium\", \"low\"\n"
    "- color: short tailwind class or hex color string light colors\n"
    "- description: short summary (1-2 sentences max)\n"
    "- criteria: short rationale for membership (single sentence)\n"
    "- commonErrors: array of short strings (top error themes)\n"
    "- suggestions: array of short actionable suggestions for the group\n"
    "- students: array of student objects, each with keys: { analysisId: string, studentName: string, rationale: string }\n"
    "CONSTRAINTS:\n"
    "- Return between 2 and 6 groups.\n"
    "- A student may appear in multiple groups if they match multiple criteria (use the analysis ID as unique student reference). Do NOT duplicate the same student more than once within a single group's students array.\n"
    "- If you are uncertain about a student's difficulty, put them in \"medium\".\n"
    "- Keep outputs concise: descriptions <= 200 chars, rationale <= 160 chars, at most 10 commonErrors and 10 suggestions per group.\n"
    "- Prefer balanced groups when reasonable, but prioritize pedagogical coherence.\n"
    "ERROR HANDLING: If you cannot produce a valid grouping, return {\"groups\":[]} as the entire response.\n"
    "EXAMPLE OUTPUT:\n"
    "{\n  \"groups\": [\n    {\n      \"id\": \"needs-support\",\n      \"name\": \"Support Group\",\n      \"level\": \"low\",\n      \"color\": \"#fee2e2\",\n      \"description\": \"Students needing targeted support on fundamentals.\",\n      \"criteria\": \"High error rates on fraction addition and missing place-value concepts.\",\n      \"commonErrors\": [\"incorrect fraction simplification\", \"misaligned place values\"],\n      \"suggestions\": [\"small-group review lesson\", \"workbook exercises\"],\n      \"students\": [\n        {\"analysisId\": \"a1\", \"studentName\": \"Maria\", \"rationale\": \"Consistent errors in fraction addition\"}\n      ]\n    }\n  ]\n}\n"
)

CACHE_TTL_SECONDS = int(float(__import__('os').getenv('GROUPING_CACHE_TTL', '120')))
_CACHE: Dict[str, Dict[str, Any]] = {}

def _cache_key(analyses: List[Dict[str, Any]]) -> str:
    # Use hash of IDs + mainError + errorPercentage for invalidation
    basis = ";".join(f"{a['id']}|{a['data']['mainError']}|{a['data']['errorPercentage']}" for a in analyses)
    return hashlib.sha256(basis.encode()).hexdigest()

def _store_cache(key: str, data: Dict[str, Any]):
    _CACHE[key] = {"data": data, "ts": time.time()}

def _get_cache(key: str):
    entry = _CACHE.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL_SECONDS:
        return None
    return entry["data"]

def fallback_groups(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Group by student first to avoid duplicates
    student_analyses = _group_analyses_by_student(analyses)
    unique_analyses = list(student_analyses.values())
    
    # Simple heuristic grouping by error range
    buckets = {"high": [], "medium": [], "low": []}
    for a in unique_analyses:
        pct = a['data'].get('errorPercentage', 50)
        if pct < 30:
            buckets['high'].append(a)
        elif pct < 60:
            buckets['medium'].append(a)
        else:
            buckets['low'].append(a)
    groups = []
    mapping = {
        'high': ("advanced", "Advanced Group", "bg-green-50 border-green-200"),
        'medium': ("intermediate", "Intermediate Group", "bg-yellow-50 border-yellow-200"),
        'low': ("needs-support", "Support Group", "bg-red-50 border-red-200")
    }
    for level, items in buckets.items():
        if not items:
            continue
        slug, name, color = mapping[level]
        groups.append({
            "id": slug,
            "name": name,
            "level": level,
            "color": color,
            "description": "Heuristic grouping (LLM disabled)",
            "criteria": f"errorPercentage bucket {level}",
            "commonErrors": list({a['data']['mainError'] for a in items})[:5],
            "suggestions": ["Enable GEMINI_API_KEY for richer grouping"],
            # Provide both analysis id and student name so frontend can show readable names
            "students": [
                {
                    "analysisId": a.get('id'),
                    "studentName": a.get('studentName', 'Unknown Student'),
                    "rationale": "heuristic"
                }
                for a in items
            ]
        })
    return {"groups": groups, "llm": False, "cached": False}

def _group_analyses_by_student(analyses: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group analyses by student, keeping the most recent analysis as representative"""
    students = {}
    for a in analyses:
        student_name = a.get('studentName', 'Unknown Student')
        if student_name not in students:
            students[student_name] = a
        else:
            # Keep the most recent analysis (assuming analyses are ordered by timestamp)
            current_timestamp = students[student_name].get('timestamp', '')
            new_timestamp = a.get('timestamp', '')
            if new_timestamp > current_timestamp:
                students[student_name] = a
    return students

def build_groups(analyses: List[Dict[str, Any]], force: bool = False) -> Dict[str, Any]:
    if not analyses:
        return {"groups": [], "llm": llm_available(), "cached": False}
    
    # Group analyses by student to avoid duplicates
    student_analyses = _group_analyses_by_student(analyses)
    unique_analyses = list(student_analyses.values())
    
    key = _cache_key(unique_analyses)
    if not force:
        cached = _get_cache(key)
        if cached:
            cached['cached'] = True
            return cached
    if not llm_available():
        data = fallback_groups(unique_analyses)
        _store_cache(key, data)
        return data
    # Build compact context with unique students
    lines = []
    for a in unique_analyses[:120]:  # limit
        lines.append(
            f"ID={a['id']}|student={a['studentName']}|subject={a['subject']}|error%={a['data']['errorPercentage']}|mainError={a['data']['mainError']}|concepts={','.join(a['data'].get('concepts', [])[:4])}"
        )
    prompt = (
        "Analysis data (one per line, each representing a unique student):\n" + "\n".join(lines) +
        "\nGenerate JSON with 'groups' key. Grouping is by student characteristics; a student may appear in multiple groups when appropriate. Do NOT duplicate the same student more than once within a single group's students array. See system instructions."
    )
    # Request a larger token budget for grouping results (many students/groups)
    result = generate_structured(prompt, system=GROUP_SYSTEM, temperature=0.25, max_retries=2, max_output_tokens=5048)
    if not result.get('success') or 'groups' not in result['data']:
        data = fallback_groups(unique_analyses)
        _store_cache(key, data)
        return data
    groups = result['data']['groups']
    # Light sanitization
    norm_groups = []
    for g in groups[:10]:
        norm_groups.append({
            "id": str(g.get('id', 'group'))[:40],
            "name": str(g.get('name', 'Group'))[:80],
            "level": g.get('level', 'medium'),
            "color": g.get('color', 'bg-gray-50 border-gray-200')[:80],
            "description": str(g.get('description', ''))[:300],
            "criteria": str(g.get('criteria', ''))[:200],
            "commonErrors": [str(e)[:120] for e in g.get('commonErrors', [])][:10],
            "suggestions": [str(s)[:160] for s in g.get('suggestions', [])][:10],
            "students": [
                {
                    "analysisId": str(s.get('id', '')),
                    "studentName": str(s.get('studentName', 'Unknown Student')),
                    "rationale": str(s.get('rationale', ''))[:160]
                }
                for s in g.get('students', [])[:200]
            ]
        })
    payload = {"groups": norm_groups, "llm": True, "cached": False}
    _store_cache(key, payload)
    return payload


# -------------------- Class-level insights --------------------
CLASS_SYSTEM = (
    "You are a pedagogical analyst. Given compact per-student analysis lines, produce a JSON object with keys:\n"
    "class_name (string), student_count (int), average_error (float), commonErrors (array of short strings), suggestions (array of short actionable items), detailed (array of objects with studentName, analysisId, errorPercentage, shortRationale).\n"
    "STRICT: Return ONLY valid JSON (no explanatory text). Keep arrays limited to top 8 items."
)


def _cache_key_for_class(analyses: List[Dict[str, Any]], class_name: str) -> str:
    basis = ";".join(f"{a['id']}|{a['data'].get('mainError')}|{a['data'].get('errorPercentage')}" for a in analyses)
    return f"class:{class_name}:{hashlib.sha256(basis.encode()).hexdigest()}"


def build_class_insights(analyses: List[Dict[str, Any]], class_name: str, force: bool = False) -> Dict[str, Any]:
    """Generate class-level insights using the LLM (cached)."""
    if not analyses:
        # Return empty but informative structure
        return {"class_name": class_name, "student_count": 0, "average_error": 0.0, "commonErrors": [], "suggestions": [], "detailed": [], "llm": llm_available(), "cached": False}

    # Use hash key
    key = _cache_key_for_class(analyses, class_name)
    if not force:
        cached = _get_cache(key)
        if cached:
            cached['cached'] = True
            return cached

    # If LLM not available, produce heuristic summary
    if not llm_available():
        # simple aggregates
        errs = []
        total = 0.0
        count = 0
        main_errors = {}
        for a in analyses:
            data = a.get('data', {})
            pct = 0.0
            try:
                pct = float(data.get('errorPercentage') or 0)
            except Exception:
                pct = 0.0
            total += pct
            count += 1
            me = data.get('mainError') or 'Unknown'
            main_errors[me] = main_errors.get(me, 0) + 1
        avg = round((total / count) if count else 0.0, 2)
        sorted_errors = sorted(main_errors.items(), key=lambda x: -x[1])[:8]
        common = [e for e, _ in sorted_errors]
        suggestions = ["Review common mistakes in class; use small-group exercises"]
        detailed = []
        for a in analyses[:40]:
            detailed.append({
                'studentName': a.get('studentName'),
                'analysisId': a.get('id'),
                'errorPercentage': a.get('data', {}).get('errorPercentage'),
                'shortRationale': str(a.get('data', {}).get('mainError'))[:140]
            })
        payload = {"class_name": class_name, "student_count": count, "average_error": avg, "commonErrors": common, "suggestions": suggestions, "detailed": detailed, "llm": False, "cached": False}
        _store_cache(key, payload)
        return payload

    # Build prompt lines
    lines = []
    for a in analyses[:400]:
        lines.append(f"ID={a['id']}|student={a.get('studentName')}|error%={a.get('data', {}).get('errorPercentage')}|mainError={a.get('data', {}).get('mainError')}|concepts={','.join(a.get('data', {}).get('concepts', [])[:4])}")

    prompt = "Class-level analyses (one per line):\n" + "\n".join(lines) + "\nProduce a JSON object as described in the system instruction."

    result = generate_structured(prompt, system=CLASS_SYSTEM, temperature=0.2, max_retries=2, max_output_tokens=1500)
    if not result.get('success') or not isinstance(result.get('data'), dict):
        # fallback
        payload = build_class_insights(analyses, class_name, force=True) if not llm_available() else {"class_name": class_name, "student_count": len(analyses), "average_error": 0.0, "commonErrors": [], "suggestions": [], "detailed": [], "llm": True, "cached": False}
        _store_cache(key, payload)
        return payload

    data = result['data']
    # minimal sanitization
    data['class_name'] = data.get('class_name', class_name)
    data['student_count'] = int(data.get('student_count') or len(analyses))
    try:
        data['average_error'] = float(data.get('average_error') or 0.0)
    except Exception:
        data['average_error'] = 0.0
    data['llm'] = True
    data['cached'] = False
    _store_cache(key, data)
    return data
