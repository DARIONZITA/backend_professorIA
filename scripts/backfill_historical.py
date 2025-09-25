"""Backfill script: agrega análises por estudante e preenche ai_structured.historicalAnalysis
Sem uso de LLM — opera apenas sobre campos já persistidos.

Usage (PowerShell):
  python .\scripts\backfill_historical.py

Isso irá editar o arquivo analyses_db.json no mesmo diretório (faça backup se quiser).
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSES_FILE = ROOT / 'analyses_db.json'

if not ANALYSES_FILE.exists():
    print(f"Arquivo não encontrado: {ANALYSES_FILE}")
    raise SystemExit(1)

with open(ANALYSES_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

analyses = data.get('analyses', [])
# Agrupar por (studentName, subject)
groups = defaultdict(list)
for a in analyses:
    student = a.get('studentName') or 'Unknown'
    subject = a.get('subject') or 'Unknown'
    groups[(student, subject)].append(a)

updated = 0
for (student, subject), items in groups.items():
    if not student or student == 'Unknown':
        continue
    main_errors = Counter()
    concepts = Counter()
    suggestions = Counter()
    recurring = 0
    total = 0
    for a in items:
        total += 1
        d = a.get('data', {})
        ai_struct = d.get('ai_structured') or {}
        if ai_struct.get('specificError'):
            main_errors[ai_struct.get('specificError')] += 1
        elif d.get('mainError'):
            main_errors[d.get('mainError')] += 1
        ai_analysis = d.get('ai_analysis') or {}
        for c in ai_analysis.get('concepts', []) or d.get('concepts', []) or []:
            try:
                concepts[str(c)] += 1
            except Exception:
                pass
        for s in ai_analysis.get('suggestions', []) or d.get('suggestions', []) or []:
            try:
                suggestions[str(s)] += 1
            except Exception:
                pass
        if ai_struct.get('isRecurrent'):
            recurring += 1

    top_errors = [k for k, _ in main_errors.most_common(3)]
    top_concepts = [k for k, _ in concepts.most_common(5)]
    top_suggestions = [k for k, _ in suggestions.most_common(5)]

    parts = []
    parts.append(f"Historical summary based on {total} previous analyses for {student} ({subject}).")
    if top_errors:
        parts.append(f"Most frequent issues: {', '.join(top_errors)}.")
    if top_concepts:
        parts.append(f"Related concepts often involved: {', '.join(top_concepts)}.")
    if recurring:
        parts.append(f"Detected {recurring} cases flagged as recurrent patterns.")
    if top_suggestions:
        parts.append(f"Common suggestions previously given: {', '.join(top_suggestions)}.")
    parts.append("Recommendation: focus targeted practice on the most frequent issues and review the related concepts listed above.")

    summary = " ".join(parts)

    # Aplicar summary a entradas que ainda não tenham historicalAnalysis
    for a in items:
        d = a.setdefault('data', {})
        ai_struct = d.setdefault('ai_structured', {})
        if not ai_struct.get('historicalAnalysis'):
            ai_struct['historicalAnalysis'] = summary
            updated += 1

# Escrever de volta apenas se houve mudanças
if updated:
    with open(ANALYSES_FILE, 'w', encoding='utf-8') as f:
        json.dump({'analyses': analyses}, f, ensure_ascii=False, indent=2)
    print(f"Backfill completo. Entradas atualizadas: {updated}")
else:
    print("Nenhuma entrada precisou de backfill.")
