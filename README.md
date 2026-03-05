# AI Interview Evaluation System

A **production-grade, fully local** AI system that evaluates technical interview recordings from any meeting platform (Zoom, Google Meet, Microsoft Teams, etc.).

No paid APIs required — everything runs on your machine using **Ollama + Whisper**.

---

## 📂 Project Structure

```
interview_ai_evaluator/
├── backend/
│   ├── app.py                 ← FastAPI entrypoint
│   ├── transcript_service.py  ← Whisper audio transcription
│   ├── transcript_cleaner.py  ← Text normalisation & filler removal
│   ├── qa_extractor.py        ← Question-answer pair segmentation
│   ├── evaluation_service.py  ← First-pass evaluation (Mistral)
│   ├── feedback_service.py    ← Second-pass validation & feedback (Llama3)
│   ├── scoring_engine.py      ← Weighted scoring + consistency check
│   └── ollama_client.py       ← Ollama API wrapper
├── models/                    ← (reserved for custom models)
├── data/
│   └── audio/                 ← Drop interview recordings here
├── requirements.txt
└── README.md
```

---

## ⚙️ Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | [python.org](https://python.org) |
| Ollama | [ollama.ai](https://ollama.ai) |
| FFmpeg | Required by Whisper for audio processing |

### Install FFmpeg (Windows)
```powershell
winget install Gyan.FFmpeg
```
Or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

---

## 🚀 Setup

### 1. Clone or navigate to the project
```bash
cd interview_ai_evaluator
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Ollama and download models
```bash
# Start Ollama (if not running as a service)
ollama serve

# In a new terminal, pull the required models
ollama pull mistral
ollama pull llama3
```

### 5. Start the server
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**  
Interactive docs: **http://localhost:8000/docs**

---

## 🔌 API Endpoints

### `GET /`
Health check.

**Response:**
```json
{ "status": "ok", "service": "AI Interview Evaluation System", "version": "1.0.0" }
```

---

### `POST /evaluate-audio`
Upload an interview audio file for transcription + evaluation.

**Form data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Audio file (mp3, mp4, m4a, wav, webm, ogg, flac) |
| `job_role` | string | ❌ | e.g. "Backend Engineer" |
| `whisper_model` | string | ❌ | `tiny` / `base` / `small` / `medium` / `large` (default: `base`) |

**cURL example:**
```bash
curl -X POST http://localhost:8000/evaluate-audio \
  -F "file=@data/audio/my_interview.mp3" \
  -F "job_role=Backend Engineer" \
  -F "whisper_model=base"
```

---

### `POST /evaluate-transcript`
Submit a pre-existing transcript for evaluation (no audio needed).

**JSON body:**
```json
{
  "transcript": "Interviewer: What is REST API?\nCandidate: REST API uses HTTP methods...",
  "job_role": "Backend Engineer"
}
```

**cURL example:**
```bash
curl -X POST http://localhost:8000/evaluate-transcript \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Interviewer: What is REST API?\nCandidate: REST is stateless...", "job_role": "Backend"}'
```

---

## 📊 Response Schema

```json
{
  "evaluated_at": "2026-03-05T05:28:00Z",
  "job_role": "Backend Engineer",
  "total_questions": 5,
  "scores": {
    "technical": 8.0,
    "problem_solving": 7.5,
    "communication": 8.0,
    "confidence": 7.0,
    "overall": 7.6
  },
  "qa_details": [
    {
      "question": "What is REST API?",
      "answer": "REST API uses HTTP methods like GET and POST...",
      "concepts_detected": ["HTTP", "GET", "POST", "stateless"],
      "missing_concepts": ["URI", "idempotent"],
      "scores": {
        "technical": 8, "keyword": 7.5, "validated": 8,
        "communication": 8, "confidence": 7, "final": 7.95
      },
      "feedback": {
        "strengths": ["Correctly identified HTTP methods"],
        "mistakes": ["Did not mention stateless constraint explicitly"],
        "correct_explanation": "REST is an architectural style based on stateless client-server communication using standard HTTP methods...",
        "improvement": "Study REST architectural constraints (Roy Fielding's dissertation, Chapter 5)"
      }
    }
  ],
  "summary_feedback": {
    "strengths": ["..."],
    "mistakes": ["..."],
    "improvement": "..."
  }
}
```

---

## 🧠 Scoring Rubric

| Score | Meaning |
|-------|---------|
| 10 | Expert explanation with concrete examples |
| 8  | Mostly correct, minor gaps |
| 6  | Basic understanding, missing key concepts |
| 4  | Weak or incomplete explanation |
| 2  | Incorrect or off-topic answer |

### Weighted Score Formula
```
final_score = 0.50 × llm_score
            + 0.20 × keyword_score
            + 0.20 × validated_score
            + 0.10 × communication_score
```

---

## 🏗️ Architecture

```
Audio Interview (any platform)
        ↓
Whisper Speech-to-Text
        ↓
Transcript Normalisation (cleaner)
        ↓
Q&A Pair Extraction (regex + LLM fallback)
        ↓
First Pass: Mistral Evaluation
        ↓
Second Pass: Llama3 Validation & Feedback
        ↓
Keyword Concept Matching
        ↓
Weighted Score Aggregation
        ↓
Consistency Check (2–3 runs, median if drift > 2)
        ↓
Final JSON Report
```

---

## 🛠️ Supported Audio Formats

Works with recordings from any meeting platform:
- **Zoom** — `.mp4`, `.m4a`
- **Google Meet** — `.mp4`, `.webm`
- **Microsoft Teams** — `.mp4`, `.m4a`
- **General** — `.mp3`, `.wav`, `.ogg`, `.flac`

---

## 📝 License
MIT — free to use and modify.
