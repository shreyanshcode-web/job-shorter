from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobBlueprint(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    secondary_skills: list[str] = Field(default_factory=list)
    role_type: str = "Unknown"
    seniority: str = "Unknown"
    experience: str = "Not specified"
    min_experience_years: int = 0
    industry: str = "General"
    behavioral_traits: list[str] = Field(default_factory=list)
    hiring_intent: str = "Find candidates with strong role fit."
    inferred_concepts: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    id: str
    name: str
    headline: str = ""
    location: str = ""
    education: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    resume_text: str = ""
    experience_years: float = 0
    career_metadata: dict[str, Any] = Field(default_factory=dict)
    activity_signals: dict[str, Any] = Field(default_factory=dict)
    social_signals: dict[str, Any] = Field(default_factory=dict)


class EnrichedCandidate(CandidateProfile):
    expanded_skills: list[str] = Field(default_factory=list)
    inferred_skills: list[str] = Field(default_factory=list)
    enrichment: dict[str, Any] = Field(default_factory=dict)


class ScoreBreakdown(BaseModel):
    semantic_fit: float
    skill_match: float
    experience_match: float
    activity_signals: float
    leadership: float
    culture_fit: float
    final_score: float


class RankedCandidate(BaseModel):
    candidate: EnrichedCandidate
    score: ScoreBreakdown
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    reason: str = ""


class RankRequest(BaseModel):
    job_description: str
    candidates: list[CandidateProfile] | None = None
    top_k: int = 10
    weights: dict[str, float] | None = None


class RankResponse(BaseModel):
    blueprint: JobBlueprint
    matches: list[RankedCandidate]


class CopilotRequest(BaseModel):
    question: str
    job_description: str
    matches: list[RankedCandidate] | None = None
    candidates: list[CandidateProfile] | None = None


class CopilotResponse(BaseModel):
    answer: str
    candidates: list[RankedCandidate] = Field(default_factory=list)
