# Job Shorter

Job Shorter is a lightweight talent intelligence prototype for matching candidates to job descriptions. It analyzes a JD, enriches candidate profiles with additional signals, retrieves the most relevant profiles, and returns an explainable ranked shortlist through a simple web interface.

## Features

- Parse job descriptions into a structured hiring blueprint
- Expand related skills through a lightweight knowledge graph
- Enrich candidate profiles with activity, leadership, learning, and consistency signals
- Retrieve candidates using semantic similarity
- Rank candidates with multi-factor weighted scoring
- Return score breakdowns, strengths, weaknesses, and short explanations
- Ask follow-up questions through a recruiter copilot endpoint

## How It Works

1. A recruiter enters a job description in the frontend.
2. The backend extracts required skills, secondary skills, role type, seniority, experience, and behavioral traits.
3. Candidate profiles are enriched with expanded skills and profile-derived signals.
4. Deterministic embeddings are generated for the job and candidates.
5. Candidates are retrieved using cosine similarity.
6. A ranking engine combines semantic fit, skills, experience, activity, leadership, and culture fit into a final score.
7. The API returns ranked candidates with explainable output.

## Tech Stack

### Frontend

- Next.js 15
- React 19
- TypeScript
- Tailwind CSS
- Lucide React

### Backend

- Python
- FastAPI
- Pydantic
- Uvicorn

### Ranking Pipeline

- Deterministic hashed embeddings
- Cosine similarity retrieval
- Skill graph expansion and inference
- Heuristic weighted scoring
- Explanation generation

### Infrastructure

- Docker
- Docker Compose

The compose file also includes PostgreSQL, Qdrant, and Neo4j service definitions for future extension. The current demo itself runs with in-memory seed data and self-contained backend logic.

## Repository Structure

```text
job-shorter/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── pipeline.py
│   │   ├── schemas.py
│   │   └── seed.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## API Endpoints

### `GET /health`

Returns the API health status.

### `GET /candidates`

Returns the seeded candidate profiles used by the demo.

### `POST /job-intelligence`

Converts a raw job description into a structured `JobBlueprint`.

Example:

```json
{
  "job_description": "Senior Data Scientist with experience in LLM deployment, RAG, MLOps, leadership, and 5+ years of experience."
}
```

### `POST /rank`

Ranks candidates for a job description and returns both the blueprint and ranked matches.

Example:

```json
{
  "job_description": "Senior Data Scientist with experience in LLM deployment, RAG, MLOps, leadership, and 5+ years of experience.",
  "top_k": 8
}
```

### `POST /copilot`

Answers questions about ranked results.

Example:

```json
{
  "question": "Why is the top candidate above the next one?",
  "job_description": "Senior Data Scientist with experience in LLM deployment, RAG, MLOps, leadership, and 5+ years of experience."
}
```

## Ranking Signals

The default ranking logic combines the following signals:

| Signal | Weight |
| --- | ---: |
| Semantic fit | 35% |
| Skill match | 20% |
| Experience match | 15% |
| Activity signals | 15% |
| Leadership | 10% |
| Culture fit | 5% |

These weights can be overridden through the `/rank` API.

## Run Locally

### Docker Compose

From the repository root:

```bash
docker compose up --build
```

Expected endpoints:

- Frontend: `http://127.0.0.1:3000`
- Backend: `http://127.0.0.1:8000`

### Manual Setup

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev -- --hostname 127.0.0.1 --port 3000
```

## Notes

- The current implementation uses seeded profiles from `backend/app/seed.py`.
- Most of the ranking and enrichment logic lives in `backend/app/pipeline.py`.
- The frontend dashboard is implemented in `frontend/app/page.tsx`.

## Future Work

- Connect live candidate/profile data sources
- Store embeddings in Qdrant for larger-scale retrieval
- Move graph logic to Neo4j-backed relationships
- Add stronger anomaly and profile quality checks
- Introduce learned ranking models on top of the current heuristic baseline
