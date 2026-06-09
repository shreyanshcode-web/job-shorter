from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.schemas import (
    CandidateProfile,
    CopilotResponse,
    EnrichedCandidate,
    JobBlueprint,
    RankedCandidate,
    ScoreBreakdown,
)


DEFAULT_WEIGHTS = {
    "semantic_fit": 0.35,
    "skill_match": 0.20,
    "experience_match": 0.15,
    "activity_signals": 0.15,
    "leadership": 0.10,
    "culture_fit": 0.05,
}

SKILL_ALIASES = {
    "large language models": "LLMs",
    "llm": "LLMs",
    "llms": "LLMs",
    "rag": "RAG",
    "retrieval augmented generation": "RAG",
    "retrieval-augmented generation": "RAG",
    "cuda": "CUDA",
    "gpu": "GPU Optimization",
    "gpu optimization": "GPU Optimization",
    "mlops": "MLOps",
    "machine learning operations": "MLOps",
    "vector database": "Vector Databases",
    "vector databases": "Vector Databases",
    "pinecone": "Pinecone",
    "qdrant": "Qdrant",
    "weaviate": "Weaviate",
    "langchain": "LangChain",
    "prompt engineering": "Prompt Engineering",
    "fine tuning": "Fine Tuning",
    "fine-tuning": "Fine Tuning",
    "embedding": "Embedding Models",
    "embeddings": "Embedding Models",
    "bge": "BGE Embeddings",
    "sentence transformers": "Sentence Transformers",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "learning to rank": "Learning to Rank",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "fastapi": "FastAPI",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "neo4j": "Neo4j",
    "leadership": "Leadership",
    "mentorship": "Mentorship",
}

SKILL_GRAPH = {
    "LLMs": [
        "RAG",
        "LangChain",
        "Prompt Engineering",
        "Fine Tuning",
        "Embedding Models",
        "Evaluation",
    ],
    "RAG": [
        "LangChain",
        "Vector Databases",
        "Pinecone",
        "Qdrant",
        "Weaviate",
        "Embedding Models",
        "BGE Embeddings",
        "Sentence Transformers",
    ],
    "MLOps": [
        "Docker",
        "Kubernetes",
        "Model Serving",
        "Monitoring",
        "CI/CD",
        "Triton Inference Server",
        "Ray",
    ],
    "GPU Optimization": [
        "CUDA",
        "Triton Inference Server",
        "Quantization",
        "Batching",
        "Ray",
    ],
    "AI Platform Engineer": ["LLMs", "MLOps", "RAG", "GPU Optimization", "Kubernetes"],
    "Ranking": ["Learning to Rank", "XGBoost", "LightGBM", "Semantic Search", "Reranking"],
}

ROLE_PATTERNS = [
    ("AI Platform Engineer", ["ai platform", "llm deployment", "gpu optimization", "mlops"]),
    ("Senior Data Scientist", ["senior data scientist", "data scientist"]),
    ("MLOps Engineer", ["mlops", "model serving", "deployment"]),
    ("NLP Research Engineer", ["nlp", "fine tuning", "transformers"]),
    ("Search Ranking Engineer", ["ranking", "learning to rank", "semantic search"]),
]


def canonical_skill(value: str) -> str:
    key = re.sub(r"[^a-z0-9+#. -]", " ", value.lower()).strip()
    key = re.sub(r"\s+", " ", key)
    return SKILL_ALIASES.get(key, value.strip())


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def clamp(value: float, lower: float = 0, upper: float = 100) -> float:
    return max(lower, min(upper, value))


def token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9+#.]+", text.lower()) if len(t) > 1}


def weighted_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(weights.values()) or 1
    return round(sum(scores[k] * weights.get(k, 0) for k in DEFAULT_WEIGHTS) / total_weight, 2)


def deterministic_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    tokens = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
    for token, count in Counter(tokens).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1 if digest[4] % 2 == 0 else -1
        vector[bucket] += sign * (1 + math.log(count))
    norm = math.sqrt(sum(v * v for v in vector)) or 1
    return [v / norm for v in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0
    return sum(a * b for a, b in zip(left, right))


def profile_text(candidate: CandidateProfile | EnrichedCandidate) -> str:
    parts: list[str] = [
        candidate.name,
        candidate.headline,
        candidate.location,
        " ".join(candidate.education),
        " ".join(candidate.skills),
        " ".join(candidate.projects),
        " ".join(candidate.certifications),
        candidate.resume_text,
    ]
    for value in candidate.career_metadata.values():
        if isinstance(value, list):
            parts.append(" ".join(str(v) for v in value))
        else:
            parts.append(str(value))
    return " ".join(parts)


class JobIntelligenceAgent:
    def analyze(self, job_description: str) -> JobBlueprint:
        text = job_description.lower()
        required, secondary = self._extract_skills(text)
        role_type = self._role_type(text)
        seniority = self._seniority(text)
        min_years = self._experience_years(text)
        industry = self._industry(text)
        behavioral_traits = self._behavioral_traits(text)
        inferred_concepts = sorted(KnowledgeGraph().expand_skills(required + secondary) - set(required) - set(secondary))
        hiring_intent = self._hiring_intent(role_type, seniority, required, behavioral_traits)

        return JobBlueprint(
            required_skills=required,
            secondary_skills=secondary,
            role_type=role_type,
            seniority=seniority,
            experience=f"{min_years}+ years" if min_years else "Not specified",
            min_experience_years=min_years,
            industry=industry,
            behavioral_traits=behavioral_traits,
            hiring_intent=hiring_intent,
            inferred_concepts=inferred_concepts[:12],
        )

    def _extract_skills(self, text: str) -> tuple[list[str], list[str]]:
        found: list[str] = []
        for key, canonical in sorted(SKILL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text):
                found.append(canonical)

        if "vector" in text and "database" in text:
            found.append("Vector Databases")
        if "open source" in text:
            found.append("Open Source")

        deduped = list(dict.fromkeys(found))
        hard_markers = ("need", "required", "must", "experience in", "looking for", "strong")
        required: list[str] = []
        secondary: list[str] = []

        for skill in deduped:
            normalized_skill = normalize(skill)
            index = normalize(text).find(normalized_skill)
            window = text[max(0, index - 80) : index + 120] if index >= 0 else text
            target = required if any(marker in window for marker in hard_markers) else secondary
            target.append(skill)

        if not required and deduped:
            required = deduped[:4]
            secondary = deduped[4:]

        return required[:10], [s for s in secondary if s not in required][:10]

    def _role_type(self, text: str) -> str:
        for role, patterns in ROLE_PATTERNS:
            if any(pattern in text for pattern in patterns):
                return role
        if "manager" in text or "lead" in text:
            return "Technical Leader"
        return "AI Talent"

    def _seniority(self, text: str) -> str:
        if re.search(r"\b(principal|staff|head|director)\b", text):
            return "Principal/Staff"
        if re.search(r"\b(senior|sr|lead)\b", text):
            return "Senior"
        if re.search(r"\b(junior|entry|associate)\b", text):
            return "Junior"
        return "Mid-Level"

    def _experience_years(self, text: str) -> int:
        match = re.search(r"(\d+)\s*\+?\s*(?:years|yrs)", text)
        return int(match.group(1)) if match else 0

    def _industry(self, text: str) -> str:
        if any(term in text for term in ["llm", "ai", "artificial intelligence", "machine learning"]):
            return "Artificial Intelligence"
        if "fintech" in text or "financial" in text:
            return "Fintech"
        if "hr" in text or "recruit" in text:
            return "HR Tech"
        return "General"

    def _behavioral_traits(self, text: str) -> list[str]:
        traits = []
        if any(term in text for term in ["leadership", "team lead", "manage", "manager"]):
            traits.append("Leadership")
        if any(term in text for term in ["mentor", "mentorship", "coach"]):
            traits.append("Mentorship")
        if any(term in text for term in ["ownership", "owner", "independent"]):
            traits.append("Ownership")
        if any(term in text for term in ["collaborate", "cross-functional", "stakeholder"]):
            traits.append("Collaboration")
        return traits

    def _hiring_intent(
        self,
        role_type: str,
        seniority: str,
        required: list[str],
        behavioral_traits: list[str],
    ) -> str:
        skill_phrase = ", ".join(required[:4]) if required else "the core role skills"
        trait_phrase = ", ".join(behavioral_traits) if behavioral_traits else "strong delivery judgment"
        return f"Find a {seniority} {role_type} with evidence of {skill_phrase} and {trait_phrase}."


class KnowledgeGraph:
    def __init__(self) -> None:
        self.graph = SKILL_GRAPH
        self.reverse_graph: dict[str, set[str]] = {}
        for parent, children in self.graph.items():
            for child in children:
                self.reverse_graph.setdefault(child, set()).add(parent)

    def expand_skills(self, skills: list[str]) -> set[str]:
        expanded = {canonical_skill(skill) for skill in skills}
        frontier = list(expanded)
        while frontier:
            current = frontier.pop()
            for neighbor in self.graph.get(current, []):
                if neighbor not in expanded:
                    expanded.add(neighbor)
                    frontier.append(neighbor)
            for parent in self.reverse_graph.get(current, set()):
                if parent not in expanded:
                    expanded.add(parent)
                    frontier.append(parent)
        return expanded

    def infer_parent_skills(self, skills: list[str]) -> set[str]:
        canonical = {canonical_skill(skill) for skill in skills}
        inferred: set[str] = set()
        for parent, children in self.graph.items():
            evidence = canonical.intersection(children)
            if len(evidence) >= 2:
                inferred.add(parent)
        return inferred - canonical


class CandidateEnrichmentLayer:
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    def enrich(self, candidate: CandidateProfile) -> EnrichedCandidate:
        inferred = sorted(self.graph.infer_parent_skills(candidate.skills + candidate.projects))
        expanded = sorted(self.graph.expand_skills(candidate.skills + inferred))
        activity = self._activity_score(candidate.activity_signals)
        leadership = self._leadership_score(candidate)
        learning = self._learning_score(candidate.activity_signals)
        innovation = self._innovation_score(candidate.activity_signals)
        consistency = self._consistency_score(candidate)

        return EnrichedCandidate(
            **candidate.model_dump(),
            expanded_skills=expanded,
            inferred_skills=inferred,
            enrichment={
                "activity_score": round(activity, 2),
                "leadership_score": round(leadership, 2),
                "learning_score": round(learning, 2),
                "innovation_score": round(innovation, 2),
                "consistency_score": round(consistency, 2),
            },
        )

    def _activity_score(self, signals: dict[str, Any]) -> float:
        commits = min(float(signals.get("github_commits_12m", 0)) / 600, 1) * 30
        stars = min(float(signals.get("github_stars", 0)) / 800, 1) * 20
        oss = min(float(signals.get("open_source_contributions", 0)) / 15, 1) * 20
        writing = min(float(signals.get("blog_posts", 0)) / 8, 1) * 12
        forums = min(float(signals.get("forum_answers", 0)) / 50, 1) * 8
        kaggle = min(float(signals.get("kaggle_medals", 0)) / 3, 1) * 5
        hackathons = min(float(signals.get("hackathons", 0)) / 4, 1) * 5
        return clamp(commits + stars + oss + writing + forums + kaggle + hackathons)

    def _leadership_score(self, candidate: CandidateProfile) -> float:
        roles = candidate.career_metadata.get("leadership_roles", [])
        team_size = float(candidate.career_metadata.get("team_size", 0) or 0)
        social = candidate.social_signals
        role_score = min(len(roles) / 3, 1) * 35
        team_score = min(team_size / 8, 1) * 35
        mentorship = 15 if social.get("mentorship") else 0
        talks = min(float(social.get("conference_talks", 0)) / 4, 1) * 15
        return clamp(role_score + team_score + mentorship + talks)

    def _learning_score(self, signals: dict[str, Any]) -> float:
        certs = min(float(signals.get("recent_certifications", 0)) / 2, 1) * 45
        skills = min(float(signals.get("new_skills_added", 0)) / 5, 1) * 55
        return clamp(certs + skills)

    def _innovation_score(self, signals: dict[str, Any]) -> float:
        patents = min(float(signals.get("patents", 0)) / 2, 1) * 30
        research = min(float(signals.get("research_papers", 0)) / 3, 1) * 35
        oss = min(float(signals.get("open_source_contributions", 0)) / 12, 1) * 20
        hackathons = min(float(signals.get("hackathons", 0)) / 4, 1) * 15
        return clamp(patents + research + oss + hackathons)

    def _consistency_score(self, candidate: CandidateProfile) -> float:
        switches = float(candidate.career_metadata.get("job_switches", 0) or 0)
        promotion = float(candidate.career_metadata.get("promotion_velocity", 0) or 0)
        stability = max(0, 1 - switches / max(candidate.experience_years, 1)) * 55
        return clamp(stability + promotion * 45)


class EmbeddingGenerationLayer:
    def embed_job(self, blueprint: JobBlueprint, job_description: str) -> list[float]:
        text = " ".join(
            [
                job_description,
                blueprint.role_type,
                blueprint.seniority,
                blueprint.industry,
                " ".join(blueprint.required_skills),
                " ".join(blueprint.secondary_skills),
                " ".join(blueprint.behavioral_traits),
                " ".join(blueprint.inferred_concepts),
            ]
        )
        return deterministic_embedding(text)

    def embed_candidate(self, candidate: EnrichedCandidate) -> list[float]:
        return deterministic_embedding(profile_text(candidate) + " " + " ".join(candidate.expanded_skills))


class RetrievalEngine:
    def __init__(self, embeddings: EmbeddingGenerationLayer) -> None:
        self.embeddings = embeddings

    def retrieve(
        self,
        job_vector: list[float],
        candidates: list[EnrichedCandidate],
        top_k: int = 100,
    ) -> list[tuple[EnrichedCandidate, float]]:
        scored = [
            (candidate, clamp((cosine(job_vector, self.embeddings.embed_candidate(candidate)) + 1) * 50))
            for candidate in candidates
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


class AIRankingEngine:
    def rank(
        self,
        blueprint: JobBlueprint,
        candidate: EnrichedCandidate,
        semantic_fit: float,
        weights: dict[str, float] | None = None,
    ) -> RankedCandidate:
        weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        scores = {
            "semantic_fit": semantic_fit,
            "skill_match": self._skill_match(blueprint, candidate),
            "experience_match": self._experience_match(blueprint, candidate),
            "activity_signals": float(candidate.enrichment.get("activity_score", 0)),
            "leadership": self._leadership_match(blueprint, candidate),
            "culture_fit": self._culture_fit(blueprint, candidate),
        }
        final = weighted_score(scores, weights)
        score = ScoreBreakdown(**scores, final_score=final)
        strengths, weaknesses = ExplainabilityAgent().explain(blueprint, candidate, score)

        reason = (
            f"{candidate.name} scores {final:.0f}/100 with "
            f"{scores['skill_match']:.0f} skill fit, {scores['experience_match']:.0f} experience fit, "
            f"and {scores['activity_signals']:.0f} activity signal strength."
        )

        return RankedCandidate(
            candidate=candidate,
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            reason=reason,
        )

    def _skill_match(self, blueprint: JobBlueprint, candidate: EnrichedCandidate) -> float:
        candidate_skills = {normalize(skill) for skill in candidate.expanded_skills}
        candidate_text = normalize(profile_text(candidate))

        def has_skill(skill: str) -> bool:
            normalized = normalize(skill)
            return normalized in candidate_skills or normalized in candidate_text

        required = blueprint.required_skills or blueprint.inferred_concepts[:4]
        secondary = blueprint.secondary_skills
        required_hits = sum(1 for skill in required if has_skill(skill))
        secondary_hits = sum(1 for skill in secondary if has_skill(skill))

        required_score = (required_hits / max(len(required), 1)) * 75
        secondary_score = (secondary_hits / max(len(secondary), 1)) * 25 if secondary else 15
        inferred_bonus = min(len(candidate.inferred_skills), 3) * 3
        return clamp(required_score + secondary_score + inferred_bonus)

    def _experience_match(self, blueprint: JobBlueprint, candidate: EnrichedCandidate) -> float:
        if blueprint.min_experience_years <= 0:
            return clamp(70 + min(candidate.experience_years, 8) * 3)
        ratio = candidate.experience_years / max(blueprint.min_experience_years, 1)
        return clamp(ratio * 85 + min(candidate.experience_years - blueprint.min_experience_years, 4) * 4)

    def _leadership_match(self, blueprint: JobBlueprint, candidate: EnrichedCandidate) -> float:
        base = float(candidate.enrichment.get("leadership_score", 0))
        if "Leadership" in blueprint.behavioral_traits or blueprint.seniority in {"Senior", "Principal/Staff"}:
            return base
        return clamp(base * 0.75 + 20)

    def _culture_fit(self, blueprint: JobBlueprint, candidate: EnrichedCandidate) -> float:
        consistency = float(candidate.enrichment.get("consistency_score", 0))
        learning = float(candidate.enrichment.get("learning_score", 0))
        trait_score = 0
        traits = {normalize(trait) for trait in blueprint.behavioral_traits}
        text = normalize(profile_text(candidate))
        if "leadership" in traits and ("lead" in text or "manager" in text or "mentor" in text):
            trait_score += 25
        if "mentorship" in traits and ("mentor" in text or candidate.social_signals.get("mentorship")):
            trait_score += 25
        if "ownership" in traits and ("owner" in text or "founding" in text):
            trait_score += 20
        if "collaboration" in traits and ("stakeholder" in text or "crossfunctional" in text):
            trait_score += 20
        if not traits:
            trait_score = 35
        return clamp(consistency * 0.35 + learning * 0.30 + trait_score)


class ExplainabilityAgent:
    def explain(
        self,
        blueprint: JobBlueprint,
        candidate: EnrichedCandidate,
        score: ScoreBreakdown,
    ) -> tuple[list[str], list[str]]:
        candidate_skills = {normalize(skill): skill for skill in candidate.expanded_skills}
        direct_hits = [
            skill for skill in blueprint.required_skills if normalize(skill) in candidate_skills
        ]
        inferred_hits = [
            skill for skill in blueprint.required_skills if skill in candidate.inferred_skills
        ]
        strengths: list[str] = []
        weaknesses: list[str] = []

        if direct_hits:
            strengths.append(f"Direct evidence for {', '.join(direct_hits[:4])}.")
        if inferred_hits:
            strengths.append(f"Knowledge graph infers {', '.join(inferred_hits[:3])} from adjacent skills.")
        if score.activity_signals >= 70:
            strengths.append("Strong public activity across code, writing, or community signals.")
        if score.leadership >= 70:
            strengths.append("Clear leadership signal through team ownership, mentorship, or senior roles.")
        if candidate.experience_years >= blueprint.min_experience_years:
            strengths.append(f"{candidate.experience_years:g} years of experience meets the role threshold.")
        if not strengths:
            strengths.append("Relevant profile, but evidence is concentrated in fewer signals.")

        missing_required = [
            skill for skill in blueprint.required_skills if normalize(skill) not in candidate_skills
        ]
        if missing_required:
            weaknesses.append(f"Limited explicit evidence for {', '.join(missing_required[:4])}.")
        if blueprint.min_experience_years and candidate.experience_years < blueprint.min_experience_years:
            weaknesses.append("Experience is below the requested minimum.")
        if score.activity_signals < 45:
            weaknesses.append("External activity signals are lighter than top-ranked peers.")
        if score.leadership < 45 and "Leadership" in blueprint.behavioral_traits:
            weaknesses.append("Leadership evidence is present but not deep enough for a senior mandate.")
        if not weaknesses:
            weaknesses.append("No major weakness detected from available data.")

        return strengths[:5], weaknesses[:4]


class CandidateCopilot:
    def answer(self, question: str, matches: list[RankedCandidate]) -> CopilotResponse:
        query = question.lower()
        if not matches:
            return CopilotResponse(answer="No ranked candidates are available yet.", candidates=[])

        if "hidden" in query or "gem" in query:
            hidden = [
                match for match in matches
                if match.score.final_score >= 72 and match.score.semantic_fit < 70 and match.score.activity_signals >= 65
            ][:3]
            if hidden:
                names = ", ".join(match.candidate.name for match in hidden)
                return CopilotResponse(answer=f"Hidden gems: {names}. They have strong activity or skill evidence despite lower semantic similarity.", candidates=hidden)
            return CopilotResponse(answer="No hidden gems crossed the current confidence threshold.", candidates=[])

        if "rag" in query and "not langchain" in query:
            filtered = [
                match for match in matches
                if "rag" in {normalize(skill) for skill in match.candidate.expanded_skills}
                and "langchain" not in {normalize(skill) for skill in match.candidate.skills}
            ][:5]
            names = ", ".join(match.candidate.name for match in filtered) or "None"
            return CopilotResponse(answer=f"Candidates with RAG evidence but no explicit LangChain: {names}.", candidates=filtered)

        if "above" in query or "compare" in query:
            top = matches[0]
            runner_up = matches[1] if len(matches) > 1 else None
            if runner_up:
                delta = top.score.final_score - runner_up.score.final_score
                answer = (
                    f"{top.candidate.name} ranks above {runner_up.candidate.name} by {delta:.1f} points. "
                    f"The edge comes from skill fit {top.score.skill_match:.0f} vs {runner_up.score.skill_match:.0f}, "
                    f"activity {top.score.activity_signals:.0f} vs {runner_up.score.activity_signals:.0f}, "
                    f"and leadership {top.score.leadership:.0f} vs {runner_up.score.leadership:.0f}."
                )
                return CopilotResponse(answer=answer, candidates=[top, runner_up])

        if "similar" in query:
            target = self._find_named_candidate(query, matches)
            if target:
                target_vector = deterministic_embedding(profile_text(target.candidate))
                similar = sorted(
                    [match for match in matches if match.candidate.id != target.candidate.id],
                    key=lambda match: cosine(target_vector, deterministic_embedding(profile_text(match.candidate))),
                    reverse=True,
                )[:3]
                names = ", ".join(match.candidate.name for match in similar)
                return CopilotResponse(answer=f"Candidates most similar to {target.candidate.name}: {names}.", candidates=similar)

        top_three = matches[:3]
        names = ", ".join(f"{match.candidate.name} ({match.score.final_score:.0f})" for match in top_three)
        return CopilotResponse(answer=f"Top recommendations: {names}. Ask for comparisons, hidden gems, or specific skill filters.", candidates=top_three)

    def _find_named_candidate(self, query: str, matches: list[RankedCandidate]) -> RankedCandidate | None:
        for match in matches:
            name_tokens = match.candidate.name.lower().split()
            if any(token in query for token in name_tokens):
                return match
        return None


@dataclass
class TalentIntelligencePipeline:
    job_agent: JobIntelligenceAgent
    graph: KnowledgeGraph
    enrichment: CandidateEnrichmentLayer
    embeddings: EmbeddingGenerationLayer
    retrieval: RetrievalEngine
    ranker: AIRankingEngine
    copilot: CandidateCopilot

    @classmethod
    def create(cls) -> "TalentIntelligencePipeline":
        graph = KnowledgeGraph()
        embeddings = EmbeddingGenerationLayer()
        return cls(
            job_agent=JobIntelligenceAgent(),
            graph=graph,
            enrichment=CandidateEnrichmentLayer(graph),
            embeddings=embeddings,
            retrieval=RetrievalEngine(embeddings),
            ranker=AIRankingEngine(),
            copilot=CandidateCopilot(),
        )

    def analyze_job(self, job_description: str) -> JobBlueprint:
        return self.job_agent.analyze(job_description)

    def rank_candidates(
        self,
        job_description: str,
        candidates: list[CandidateProfile],
        top_k: int = 10,
        weights: dict[str, float] | None = None,
    ) -> tuple[JobBlueprint, list[RankedCandidate]]:
        blueprint = self.analyze_job(job_description)
        enriched = [self.enrichment.enrich(candidate) for candidate in candidates]
        job_vector = self.embeddings.embed_job(blueprint, job_description)
        retrieved = self.retrieval.retrieve(job_vector, enriched, top_k=100)
        ranked = [
            self.ranker.rank(blueprint, candidate, semantic_fit=semantic_fit, weights=weights)
            for candidate, semantic_fit in retrieved
        ]
        ranked.sort(key=lambda match: match.score.final_score, reverse=True)
        return blueprint, ranked[:top_k]
