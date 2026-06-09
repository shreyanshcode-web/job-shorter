from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline import TalentIntelligencePipeline
from app.schemas import (
    CandidateProfile,
    CopilotRequest,
    CopilotResponse,
    JobBlueprint,
    RankRequest,
    RankResponse,
)
from app.seed import SEED_CANDIDATES

app = FastAPI(title="Talent Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = TalentIntelligencePipeline.create()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/candidates", response_model=list[CandidateProfile])
def candidates() -> list[CandidateProfile]:
    return SEED_CANDIDATES


@app.post("/job-intelligence", response_model=JobBlueprint)
def job_intelligence(payload: dict[str, str]) -> JobBlueprint:
    job_description = payload.get("job_description", "").strip()
    if not job_description:
        raise HTTPException(status_code=400, detail="job_description is required")
    return pipeline.analyze_job(job_description)


@app.post("/rank", response_model=RankResponse)
def rank(payload: RankRequest) -> RankResponse:
    if not payload.job_description.strip():
        raise HTTPException(status_code=400, detail="job_description is required")

    candidate_pool = payload.candidates or SEED_CANDIDATES
    blueprint, matches = pipeline.rank_candidates(
        job_description=payload.job_description,
        candidates=candidate_pool,
        top_k=max(1, min(payload.top_k, 100)),
        weights=payload.weights,
    )
    return RankResponse(blueprint=blueprint, matches=matches)


@app.post("/copilot", response_model=CopilotResponse)
def copilot(payload: CopilotRequest) -> CopilotResponse:
    if payload.matches:
        matches = payload.matches
    else:
        candidate_pool = payload.candidates or SEED_CANDIDATES
        _, matches = pipeline.rank_candidates(
            job_description=payload.job_description,
            candidates=candidate_pool,
            top_k=10,
        )
    return pipeline.copilot.answer(payload.question, matches)
