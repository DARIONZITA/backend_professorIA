"""Database utilities for the Professor Assistence backend.

This module centralizes the MongoDB connection and exposes helper functions
for working with students and analyses collections.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import re
from pymongo import ASCENDING, DESCENDING, MongoClient
import certifi
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

log = logging.getLogger("db")

_client_lock: Lock = Lock()
_client: Optional[MongoClient] = None
_db_name: str = os.getenv("MONGO_DB_NAME", "professorai")
_students_col: Optional[Collection] = None
_analyses_col: Optional[Collection] = None

_DEFAULT_STUDENTS = [
    {"id": "d6133d2a-d708-4110-b994-b8fdb8b38649", "name": "Anna Smith", "class_name": "5th A"},
    {"id": "d20743a8-ab9e-454f-a2ea-43b15141675e", "name": "Bruno Johnson", "class_name": "5th A"},
    {"id": "1d426497-ecdd-46f2-b268-63d2d55d9609", "name": "Carla Williams", "class_name": "5th A"},
    {"id": "a9ae0072-3700-400f-a0c9-d8366b0796c3", "name": "Daniel Brown", "class_name": "5th A"},
    {"id": "0f426d1b-1551-4736-b205-87fb34fcddfd", "name": "Elena Davis", "class_name": "5th A"},
    {"id": "e15c33ae-6827-4def-9a07-7c819b569065", "name": "Felix Miller", "class_name": "5th A"},
    {"id": "ce6991e6-81d4-4976-a98d-bad4c3279d5c", "name": "Gabriela Wilson", "class_name": "5th A"},
    {"id": "f25a13cb-70df-446e-bf6c-15b3e3a87b5a", "name": "Hugo Garcia", "class_name": "5th A"},
]


def _load_dotenv_if_present() -> None:
    """Load environment variables from .env when possible."""
    if os.getenv("MONGO_URI"):
        return
    try:
        import dotenv  # type: ignore

        if dotenv.load_dotenv():
            log.info("Loaded environment variables from .env using python-dotenv")
            return
    except Exception:
        pass

    # Manual fallback search for .env
    for candidate in (Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
        env_path = candidate / ".env"
        if env_path.exists():
            try:
                with env_path.open("r", encoding="utf-8") as handle:
                    for raw in handle:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value and key not in os.environ:
                            os.environ[key] = value
                log.info("Loaded environment variables manually from %s", env_path)
                return
            except Exception:
                continue


_load_dotenv_if_present()


def init_db() -> None:
    """Initialize the MongoDB client, collections and indexes."""
    global _client, _students_col, _analyses_col

    if _client is not None:
        return

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI environment variable is not defined.")

    with _client_lock:
        if _client is not None:
            return
        try:
            client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=10000,
                tlsCAFile=certifi.where(),
            )
            # Force connection (raises if credentials/URI invalid)
            client.admin.command("ping")
        except PyMongoError as exc:
            raise RuntimeError(f"Failed to connect to MongoDB: {exc}") from exc

        db = client[_db_name]
        students_col = db["students"]
        analyses_col = db["analyses"]

        # Create indexes for faster lookups
        students_col.create_index("id", unique=True)
        students_col.create_index("name")

        analyses_col.create_index("id", unique=True)
        analyses_col.create_index([("studentName", ASCENDING), ("timestamp", DESCENDING)])
        analyses_col.create_index("subject")

        _client = client
        _students_col = students_col
        _analyses_col = analyses_col

        _auto_import_json_if_needed()
        _ensure_default_students()


def _auto_import_json_if_needed() -> None:
    """Import existing local JSON data into MongoDB if collections are empty."""
    if _students_col is None or _analyses_col is None:
        return

    if _students_col.count_documents({}) == 0:
        _import_students_from_json()

    if _analyses_col.count_documents({}) == 0:
        _import_analyses_from_json()


def _import_students_from_json() -> None:
    json_path = Path("student_db.json")
    if not json_path.exists():
        return
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        students = raw.get("students", [])
        if students:
            _students_col.insert_many(students, ordered=False)
            log.info("Imported %d students from %s", len(students), json_path)
    except Exception as exc:
        log.warning("Failed to import students from %s: %s", json_path, exc)


def _import_analyses_from_json() -> None:
    json_path = Path("analyses_db.json")
    if not json_path.exists():
        return
    try:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        analyses = raw.get("analyses", [])
        if analyses:
            _analyses_col.insert_many(analyses, ordered=False)
            log.info("Imported %d analyses from %s", len(analyses), json_path)
    except Exception as exc:
        log.warning("Failed to import analyses from %s: %s", json_path, exc)


def _ensure_default_students() -> None:
    if _students_col is None:
        return
    if _students_col.count_documents({}) > 0:
        return
    try:
        _students_col.insert_many(_DEFAULT_STUDENTS, ordered=False)
        log.info("Seeded default students into MongoDB")
    except Exception as exc:
        log.warning("Failed to seed default students: %s", exc)


def _ensure_initialized() -> None:
    if _client is None:
        init_db()


def _sanitize(document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if document is None:
        return None
    sanitized = dict(document)
    sanitized.pop("_id", None)
    return sanitized


# ---------------------- Student helpers ----------------------

def list_students(class_name: Optional[str] = None) -> List[Dict[str, Any]]:
    _ensure_initialized()
    if _students_col is None:
        return []
    query: Dict[str, Any] = {}
    if class_name:
        query["class_name"] = class_name
    docs = _students_col.find(query, {"_id": 0}).sort("name", ASCENDING)
    return list(docs)


def get_student_by_id(student_id: str) -> Optional[Dict[str, Any]]:
    _ensure_initialized()
    if _students_col is None:
        return None
    doc = _students_col.find_one({"id": student_id}, {"_id": 0})
    return doc


def create_student(student: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_initialized()
    if _students_col is None:
        raise RuntimeError("Students collection is not available")
    # Check for duplicate student by name + class_name (case-insensitive)
    try:
        name_val = student.get("name", "")
        class_val = student.get("class_name", "")
        if name_val:
            existing = _students_col.find_one({
                "name": {"$regex": f'^{re.escape(name_val)}$', "$options": "i"},
                "class_name": {"$regex": f'^{re.escape(class_val)}$', "$options": "i"}
            })
            if existing:
                # Raise a ValueError which the API layer will translate to HTTP 409
                raise ValueError("Student with the same name and class already exists")
    except ValueError:
        # Re-raise ValueError for duplicate detection to be handled by the caller
        raise
    except Exception:
        # Any other errors during the duplicate check should not block normal insertion;
        # we'll log and continue with insertion. Avoid failing the whole flow for
        # transient query issues.
        pass

    # Insert and return the stored document fetched from the DB to avoid in-memory
    # objects that might contain non-serializable types (e.g. ObjectId).
    _students_col.insert_one(student)
    stored = _students_col.find_one({"id": student.get("id")}, {"_id": 0})
    return _sanitize(stored) or student
def delete_student(student_id: str) -> bool:
    _ensure_initialized()
    if _students_col is None:
        return False
    try:
        res = _students_col.delete_one({"id": student_id})
        return res.deleted_count > 0
    except Exception:
        return False


# ---------------------- Analysis helpers ----------------------

def list_analyses(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    _ensure_initialized()
    if _analyses_col is None:
        return []
    cursor = _analyses_col.find({}, {"_id": 0}).sort("timestamp", DESCENDING)
    if limit:
        cursor = cursor.limit(int(limit))
    return list(cursor)


def get_analysis_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    _ensure_initialized()
    if _analyses_col is None:
        return None
    doc = _analyses_col.find_one({"id": analysis_id}, {"_id": 0})
    return doc


def insert_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_initialized()
    if _analyses_col is None:
        raise RuntimeError("Analyses collection is not available")
    _analyses_col.replace_one({"id": analysis.get("id")}, analysis, upsert=True)
    return analysis


def list_analyses_by_student(student_name: str, subject: Optional[str] = None) -> List[Dict[str, Any]]:
    _ensure_initialized()
    if _analyses_col is None:
        return []
    query: Dict[str, Any] = {"studentName": student_name}
    if subject:
        query["subject"] = subject
    cursor = _analyses_col.find(query, {"_id": 0}).sort("timestamp", DESCENDING)
    return list(cursor)


def list_analyses_by_ids(analysis_ids: List[str]) -> List[Dict[str, Any]]:
    _ensure_initialized()
    if _analyses_col is None:
        return []
    if not analysis_ids:
        return []
    cursor = _analyses_col.find({"id": {"$in": analysis_ids}}, {"_id": 0})
    return list(cursor)


def list_analyses_grouped_by_class() -> List[Dict[str, Any]]:
    """Return analyses grouped by the student's class_name.

    Output shape:
    [
      {
        'class_name': '5th A',
        'analyses': [ ... ],
        'count': 10,
        'average_error': 23.4
      },
      ...
    ]
    """
    _ensure_initialized()
    if _analyses_col is None or _students_col is None:
        return []

    try:
        # Try to leverage MongoDB aggregation to join students by name -> class_name
        pipeline = [
            {
                "$lookup": {
                    "from": "students",
                    "localField": "studentName",
                    "foreignField": "name",
                    "as": "student_docs"
                }
            },
            {
                "$unwind": {"path": "$student_docs", "preserveNullAndEmptyArrays": True}
            },
            {
                "$addFields": {"class_name": {"$ifNull": ["$student_docs.class_name", "Unknown"]}}
            },
            {
                "$group": {
                    "_id": "$class_name",
                    "analyses": {"$push": {"id": "$id", "studentName": "$studentName", "subject": "$subject", "timestamp": "$timestamp", "data": "$data"}},
                    "count": {"$sum": 1},
                    "avgError": {"$avg": {"$ifNull": ["$data.errorPercentage", 0]}}
                }
            },
            {"$project": {"class_name": "$_id", "analyses": 1, "count": 1, "average_error": {"$round": ["$avgError", 2]}, "_id": 0}},
            {"$sort": {"class_name": 1}}
        ]

        cursor = _analyses_col.aggregate(pipeline)
        return list(cursor)
    except Exception:
        # Fallback to Python grouping if aggregation fails
        analyses = list(_analyses_col.find({}, {"_id": 0}))
        # Build student name -> class map
        students = list(_students_col.find({}, {"_id": 0}))
        name_to_class: Dict[str, str] = {s.get('name'): s.get('class_name', 'Unknown') for s in students}

        groups: Dict[str, Dict[str, Any]] = {}
        for a in analyses:
            cls = name_to_class.get(a.get('studentName'), 'Unknown')
            g = groups.setdefault(cls, {'class_name': cls, 'analyses': [], 'count': 0, 'sum_error': 0.0})
            g['analyses'].append(a)
            g['count'] += 1
            try:
                g['sum_error'] += float(a.get('data', {}).get('errorPercentage') or 0)
            except Exception:
                pass

        results: List[Dict[str, Any]] = []
        for cls, val in groups.items():
            avg = (val['sum_error'] / val['count']) if val['count'] else 0.0
            results.append({'class_name': cls, 'analyses': val['analyses'], 'count': val['count'], 'average_error': round(avg, 2)})

        results.sort(key=lambda x: x['class_name'])
        return results


__all__ = [
    "init_db",
    "list_students",
    "get_student_by_id",
    "create_student",
    "list_analyses",
    "get_analysis_by_id",
    "insert_analysis",
    "list_analyses_by_student",
    "list_analyses_by_ids",
    "list_analyses_grouped_by_class",
]
