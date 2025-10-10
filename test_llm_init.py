#!/usr/bin/env python3
"""Test llm_client initialization"""
import logging
logging.basicConfig(level=logging.INFO)

# Import and test
from llm_client import llm_available, generate_structured

print("=" * 60)
print("Testing LLM Client")
print("=" * 60)

print(f"\n1. LLM Available: {llm_available()}")

if llm_available():
    print("\n2. Testing simple structured generation...")
    result = generate_structured(
        prompt="Return a JSON with keys 'test' (value: 'success') and 'number' (value: 42)",
        system="You are a helpful assistant. Return only valid JSON.",
        temperature=0.1
    )
    
    print(f"\n3. Result:")
    print(f"   Success: {result.get('success')}")
    print(f"   Error: {result.get('error')}")
    print(f"   Data: {result.get('data')}")
    print(f"   LLM Enabled: {result.get('llm_enabled')}")
else:
    print("\n✗ LLM not available - check API keys in .env")
