"""Test Groq LLM client after fix"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_client import generate_structured, llm_available

# Test simple structured generation
print("🧪 Testing Groq LLM client...")
print(f"LLM available: {llm_available()}")

if llm_available():
    print("\n📝 Testing structured JSON generation...")
    result = generate_structured(
        prompt="Analyze this text and return JSON with keys 'summary' (short text) and 'sentiment' (positive/negative/neutral): 'I love this amazing product!'",
        system="You are a helpful assistant that returns only valid JSON.",
        temperature=0.1
    )
    
    print(f"\n✅ Success: {result.get('success')}")
    print(f"📦 Data: {result.get('data')}")
    print(f"❌ Error: {result.get('error')}")
    print(f"🔧 Provider enabled: {result.get('llm_enabled')}")
else:
    print("❌ LLM not available - check API keys in .env")
