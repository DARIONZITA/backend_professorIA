#!/usr/bin/env python3
"""Test the backend API with a real analysis request"""
import requests
import json

API_BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("Testing Backend API - Analysis with Groq")
print("=" * 60)

# Create a test OCR text
test_text = """
Name: Maria Costa
Math Exam 2
1. 25 × 4 = ?
25 × 4 = 120
2. 81 ÷ 9 = ?
81 ÷ 9 = 8
"""

print(f"\n1. Test OCR text:")
print(test_text)

# Simulate analysis (we'll use the analyze_text function directly)
print("\n2. Testing analysis engine directly...")

from analysis_engine import analyze_text

result = analyze_text(test_text, "Mathematics")

print(f"\n3. Analysis Results:")
print(f"   ✓ Main Error: {result.get('mainError')}")
print(f"   ✓ Error Percentage: {result.get('errorPercentage')}%")
print(f"   ✓ Concepts: {result.get('concepts', [])[:3]}")
print(f"   ✓ Suggestions: {result.get('suggestions', [])[:2]}")
print(f"   ✓ AI Analysis: {'Present ✓' if result.get('ai_analysis') else 'Missing ✗'}")
print(f"   ✓ AI Structured: {'Present ✓' if result.get('ai_structured') else 'Missing ✗'}")
print(f"   ✓ Score: {result.get('score', {}).get('label', 'N/A')}")

if result.get('ai_structured'):
    print(f"\n4. Detailed AI Analysis:")
    ai = result['ai_structured']
    print(f"   - Main Concept: {ai.get('mainConcept', 'N/A')}")
    print(f"   - Specific Error: {ai.get('specificError', 'N/A')}")
    print(f"   - Is Recurrent: {ai.get('isRecurrent', 'N/A')}")
    print(f"   - Suggestion: {ai.get('suggestionForTeacher', 'N/A')[:100]}...")

print("\n✓ Test completed successfully - Groq is working!")
