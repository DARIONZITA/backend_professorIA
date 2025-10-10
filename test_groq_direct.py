#!/usr/bin/env python3
"""Test Groq API directly to diagnose issues"""
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()

API_GROQ = os.getenv("API_GROQ")
print(f"✓ API_GROQ loaded: {API_GROQ[:20]}..." if API_GROQ else "✗ API_GROQ not found")

if not API_GROQ:
    print("ERROR: API_GROQ not configured")
    exit(1)

# Test simple Groq request
import requests
import json

url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_GROQ}",
    "Content-Type": "application/json"
}

payload = {
    "model": "llama-3.3-70b-versatile",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Return only valid JSON."},
        {"role": "user", "content": 'Return a JSON object with keys "status" (value: "ok") and "message" (value: "Groq is working")'}
    ],
    "temperature": 0.1,
    "max_tokens": 100,
    "response_format": {"type": "json_object"}
}

print("\n🚀 Testing Groq API...")
print(f"Model: {payload['model']}")
print(f"URL: {url}")

try:
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"\n📊 Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Success!")
        print(f"\nFull response:")
        print(json.dumps(data, indent=2))
        
        # Extract content
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")
            print(f"\n📝 Content:")
            print(content)
            
            # Try parse JSON
            try:
                parsed = json.loads(content)
                print(f"\n✓ JSON parsed successfully:")
                print(json.dumps(parsed, indent=2))
            except:
                print(f"\n✗ Could not parse content as JSON")
    else:
        print(f"✗ Request failed")
        print(f"Response: {response.text[:500]}")
        
except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()
