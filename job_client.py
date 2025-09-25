#!/usr/bin/env python3
"""
Asynchronous Jobs Client for the Professor AI API.
Flow:
 1. Image upload (POST /upload)
 2. Job start informing returned path (POST /jobs/start_from_path)
 3. Polling on /jobs/{job_id} until final status

Requires: Space to be running with REST routes registered.
"""

import requests
import time
import os
import json
from typing import Optional

# Allow overriding Space URL via environment variable for local tests
SPACE_URL = os.environ.get("SPACE_URL", "https://dnzita-professorIa.hf.space")  # Adjust if necessary
IMAGE_PATH = "C:\\Users\\DELL\\OneDrive\\Documents\\hacktohns\\professor assistence\\imagem.jpg"
LANGUAGE = "English"
POLL_INTERVAL = 5  # seconds
TIMEOUT = 15 * 60   # 15 minutes


def upload_image(image_path: str) -> Optional[str]:
    print("📤 Uploading image...")
    with open(image_path, "rb") as f:
        files = {"files": f}
        r = requests.post(f"{SPACE_URL}/upload", files=files, timeout=60)
    if r.status_code != 200:
        print(f"❌ Upload failed: {r.status_code}")
        print(r.text[:200])
        return None
    try:
        data = r.json()
    except Exception:
        print("❌ Upload response was not valid JSON")
        return None

    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, str):
            print(f"✅ Upload OK: {first}")
            return first
        elif isinstance(first, dict) and 'path' in first:
            print(f"✅ Upload OK: {first['path']}")
            return first['path']
    elif isinstance(data, dict) and 'path' in data:
        print(f"✅ Upload OK: {data['path']}")
        return data['path']

    print(f"⚠️ Unexpected format in upload: {data}")
    return None


def start_job_from_server_path(path: str, language: str) -> Optional[str]:
    print("🚀 Starting asynchronous job...")
    payload = {"path": path, "language": language}
    r = requests.post(f"{SPACE_URL}/jobs/start_from_path", json=payload, timeout=30)
    if r.status_code != 200:
        print(f"❌ Failed to start job: {r.status_code}")
        print(r.text[:200])
        return None
    try:
        data = r.json()
    except Exception:
        print("❌ Invalid response when starting job")
        return None
    if 'job_id' in data:
        print(f"✅ Job started: {data['job_id']}")
        return data['job_id']
    print(f"⚠️ Unexpected response when starting: {data}")
    return None


def get_job_status(job_id: str) -> Optional[dict]:
    r = requests.get(f"{SPACE_URL}/jobs/{job_id}", timeout=30)
    if r.status_code != 200:
        print(f"⚠️ Failed to query status ({r.status_code})")
        return None
    try:
        return r.json()
    except Exception:
        return None


def poll_job(job_id: str):
    print("⏳ Waiting for job completion...")
    start = time.time()
    last_stage = None
    while True:
        if time.time() - start > TIMEOUT:
            print("⏰ Timeout waiting for job")
            return None
        status = get_job_status(job_id)
        if not status:
            print("⚠️ No status returned, trying again...")
            time.sleep(POLL_INTERVAL)
            continue

        stage = status.get('stage')
        progress = status.get('progress')
        state = status.get('status')
        if stage != last_stage:
            print(f"📍 {state} | {progress}% | {stage}")
            last_stage = stage
        else:
            print(f"… {progress}%")

        if state in ("completed", "failed", "cancelled"):
            return status
        time.sleep(POLL_INTERVAL)


def main():
    print("🧪 Asynchronous Jobs Client - Professor AI")
    print("=" * 60)
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image not found: {IMAGE_PATH}")
        return

    # 1. Upload
    server_path = upload_image(IMAGE_PATH)
    if not server_path:
        return

    # 2. Start job
    job_id = start_job_from_server_path(server_path, LANGUAGE)
    if not job_id:
        return

    # 3. Polling
    final_status = poll_job(job_id)
    if not final_status:
        print("❌ Could not get final status")
        return

    print("\n📦 FINAL STATUS:")
    print(json.dumps(final_status, indent=2, ensure_ascii=False))

    if final_status.get('status') == 'completed':
        print("\n📝 RESULT:")
        print(final_status.get('result'))
    else:
        print("\n⚠️ Job did not complete successfully.")

if __name__ == "__main__":
    main()
