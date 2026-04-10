# Professor AI — Radar Pedagógico

Backend REST para análise pedagógica de exercícios manuscritos. O serviço recebe imagens, extrai texto via OCR multimodal, analisa as dificuldades do aluno usando LLM e agrupa estudantes em perfis de aprendizagem.

---

## Stack tecnológica

| Camada | Tecnologia |
|---|---|
| API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Banco de dados | MongoDB Atlas (pymongo 4.8+) |
| OCR principal | Google Gemini 2.5 Flash (multimodal) |
| OCR legado | TrOCR via HuggingFace Space (upload/poll assíncrono) |
| LLM (preferencial) | Groq — modelo `llama-3.3-70b-versatile` |
| LLM (fallback) | Google Gemini (`gemini-1.5-flash-latest`) |
| Validação | Pydantic v2 |
| Variáveis de ambiente | python-dotenv |
| Imagens | Pillow (PIL) |
| Runtime | Python 3.11+ |

---

## Arquitetura

```
┌─────────────────────────────────────────────────────┐
│                    Cliente (Next.js / Vercel)        │
└────────────────────────┬────────────────────────────┘
                         │ HTTPS / REST
┌────────────────────────▼────────────────────────────┐
│                  main.py  (FastAPI)                  │
│  CORS · Static files (/temp) · Pydantic models      │
│                                                      │
│  /students          /analyses       /classes/{name} │
│  /analyze_exercise  /student_groups                 │
└───┬──────────────────┬──────────────────┬───────────┘
    │                  │                  │
    ▼                  ▼                  ▼
transcription_    analysis_         grouping_
engine.py         engine.py         engine.py
    │                  │                  │
    │           llm_client.py ◄───────────┘
    │           (Groq → Gemini fallback)
    │
    ├── Gemini multimodal (fast path)
    └── TrOCR via job_client.py (legacy poll)
                         │
                    db.py (MongoDB)
```

### Módulos

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Aplicação FastAPI: roteamento, upload de arquivos, orquestração OCR → análise |
| `transcription_engine.py` | Dois motores de OCR: Gemini 2.5 Flash (padrão) e TrOCR remoto (legado) |
| `analysis_engine.py` | Converte texto OCR em JSON pedagógico estruturado via LLM; fallback heurístico |
| `grouping_engine.py` | Agrupa análises em perfis de aprendizagem via LLM; cache TTL configurável |
| `llm_client.py` | Adaptador de LLM: prefere Groq, cai para Gemini; extrai JSON da resposta |
| `db.py` | Helpers MongoDB: init, coleções `students`/`analyses`, seed de dados padrão |
| `job_client.py` | Cliente assíncrono para o HuggingFace Space TrOCR (upload → start → poll) |

---

## Fluxo principal — `POST /analyze_exercise`

```
1. Recebe multipart/form-data (imagem + student_id + subject)
2. Salva imagem em temp/images/
3. transcrever_imagem_com_gemini()  →  texto OCR
4. compute_historical_summary()     →  contexto histórico do aluno
5. analyze_text()                   →  JSON pedagógico via Groq/Gemini
6. db.insert_analysis()             →  persiste no MongoDB
7. Retorna JSON com OCR + análise + URL da imagem
```

### Lógica de fallback do LLM

```
API_GROQ definido?
  ├── sim → Groq (llama-3.3-70b-versatile, endpoint OpenAI-compatible)
  └── não → GEMINI_API_KEY definido?
              ├── sim → Gemini text API
              └── não → heurística local (sem LLM)
```

---

## API — Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/students` | Lista estudantes (filtro `?class_name=`) |
| `POST` | `/students` | Cria estudante `{"name", "class_name"}` |
| `DELETE` | `/students/{id}` | Remove estudante |
| `GET` | `/classes` | Lista turmas com contagem de alunos |
| `GET` | `/classes/{class_name}` | Insights da turma via LLM (`?force=true` recomputa) |
| `POST` | `/analyze_exercise` | Upload de imagem → OCR → análise pedagógica |
| `GET` | `/analyses` | Lista todas as análises |
| `GET` | `/analyses/{id}` | Detalhe de uma análise |
| `GET` | `/student_groups` | Grupos de aprendizagem (cacheado) |
| `POST` | `/student_groups/recompute` | Recomputa agrupamentos |

---

## Variáveis de ambiente

Copie `.env.example` para `.env` e preencha os valores. **Nunca commite `.env`.**

| Variável | Obrigatória | Descrição |
|---|---|---|
| `MONGO_URI` | ✅ | Connection string MongoDB Atlas |
| `MONGO_DB_NAME` | — | Nome do banco (padrão: `professorai`) |
| `API_GROQ` | — | API key Groq (LLM preferencial) |
| `GROQ_MODEL` | — | Modelo Groq (padrão: `llama-3.3-70b-versatile`) |
| `API_GROQ_URL` | — | Endpoint Groq customizado |
| `GEMINI_API_KEY` | — | API key Google Gemini (OCR + LLM fallback) |
| `GEMINI_MODEL` | — | Modelo Gemini (padrão: `gemini-1.5-flash-latest`) |
| `SPACE_URL` | — | URL do HuggingFace Space TrOCR (OCR legado) |
| `GROUPING_CACHE_TTL` | — | TTL do cache de agrupamento em segundos (padrão: `120`) |

---

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .\.venv\Scripts\Activate.ps1    # Windows

pip install -r requirements.txt

cp .env.example .env
# edite .env com suas chaves

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

A documentação interativa estará disponível em `http://localhost:8000/docs`.

---

## Licença

MIT — ver `LICENSE`.
