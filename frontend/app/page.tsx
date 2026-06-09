"use client";

import {
  Activity,
  BrainCircuit,
  ChevronRight,
  GitBranch,
  MessageSquareText,
  Network,
  Search,
  SlidersHorizontal,
  Sparkles,
  Trophy,
  Users,
} from "lucide-react";
import { useMemo, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const defaultJob = `Senior Data Scientist

Need experience in:
- LLM deployment
- GPU optimization
- RAG
- MLOps
- Leadership
- Mentorship
- 5+ years`;

type Blueprint = {
  required_skills: string[];
  secondary_skills: string[];
  role_type: string;
  seniority: string;
  experience: string;
  industry: string;
  behavioral_traits: string[];
  hiring_intent: string;
  inferred_concepts: string[];
};

type Candidate = {
  id: string;
  name: string;
  headline: string;
  location: string;
  skills: string[];
  projects: string[];
  experience_years: number;
  expanded_skills: string[];
  inferred_skills: string[];
  enrichment: Record<string, number>;
};

type Match = {
  candidate: Candidate;
  score: {
    semantic_fit: number;
    skill_match: number;
    experience_match: number;
    activity_signals: number;
    leadership: number;
    culture_fit: number;
    final_score: number;
  };
  strengths: string[];
  weaknesses: string[];
  reason: string;
};

type RankResponse = {
  blueprint: Blueprint;
  matches: Match[];
};

const weights = [
  ["semantic_fit", "Semantic Fit", 35],
  ["skill_match", "Skill Match", 20],
  ["experience_match", "Experience", 15],
  ["activity_signals", "Activity", 15],
  ["leadership", "Leadership", 10],
  ["culture_fit", "Culture", 5],
];

function scoreColor(score: number) {
  if (score >= 85) return "bg-emerald-600";
  if (score >= 70) return "bg-sky-600";
  if (score >= 55) return "bg-amber-500";
  return "bg-zinc-500";
}

function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center rounded border border-zinc-200 bg-white px-2 py-1 text-xs font-medium text-zinc-700">
      {children}
    </span>
  );
}

function Metric({
  label,
  value,
  icon,
}: {
  label: string;
  value: number;
  icon: React.ReactNode;
}) {
  return (
    <div className="rounded border border-zinc-200 bg-white p-3">
      <div className="flex items-center gap-2 text-xs font-medium text-zinc-500">
        {icon}
        {label}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <div className="h-2 flex-1 rounded bg-zinc-100">
          <div className={`h-2 rounded ${scoreColor(value)}`} style={{ width: `${value}%` }} />
        </div>
        <span className="w-9 text-right text-sm font-semibold text-zinc-900">{Math.round(value)}</span>
      </div>
    </div>
  );
}

export default function Page() {
  const [jobDescription, setJobDescription] = useState(defaultJob);
  const [result, setResult] = useState<RankResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [question, setQuestion] = useState("Why is the top candidate above the next one?");
  const [copilotAnswer, setCopilotAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const selected = useMemo(() => {
    if (!result?.matches.length) return null;
    return result.matches.find((match) => match.candidate.id === selectedId) ?? result.matches[0];
  }, [result, selectedId]);

  async function runRanking() {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/rank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_description: jobDescription, top_k: 8 }),
      });
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as RankResponse;
      setResult(data);
      setSelectedId(data.matches[0]?.candidate.id ?? null);
      setCopilotAnswer("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ranking failed");
    } finally {
      setLoading(false);
    }
  }

  async function askCopilot() {
    if (!result) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/copilot`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          job_description: jobDescription,
          matches: result.matches,
        }),
      });
      if (!response.ok) throw new Error(await response.text());
      const data = (await response.json()) as { answer: string };
      setCopilotAnswer(data.answer);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Copilot failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-950">
      <header className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded bg-zinc-950 text-white">
              <BrainCircuit size={19} />
            </div>
            <div>
              <h1 className="text-base font-semibold tracking-normal">Talent Intelligence</h1>
              <p className="text-xs text-zinc-500">Semantic retrieval, graph inference, AI ranking</p>
            </div>
          </div>
          <button
            onClick={runRanking}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded bg-zinc-950 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-zinc-800 disabled:cursor-not-allowed disabled:bg-zinc-400"
          >
            <Search size={16} />
            {loading ? "Running" : "Run Match"}
          </button>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-5 px-5 py-5 lg:grid-cols-[380px_1fr]">
        <section className="space-y-5">
          <div className="rounded border border-zinc-200 bg-white">
            <div className="flex items-center gap-2 border-b border-zinc-200 px-4 py-3">
              <SlidersHorizontal size={16} />
              <h2 className="text-sm font-semibold">Job Intelligence</h2>
            </div>
            <div className="p-4">
              <textarea
                value={jobDescription}
                onChange={(event) => setJobDescription(event.target.value)}
                className="h-64 w-full resize-none rounded border border-zinc-300 bg-white p-3 text-sm leading-6 outline-none ring-0 focus:border-zinc-950"
              />
              {error ? <p className="mt-3 rounded border border-red-200 bg-red-50 p-3 text-xs text-red-700">{error}</p> : null}
            </div>
          </div>

          <div className="rounded border border-zinc-200 bg-white">
            <div className="flex items-center gap-2 border-b border-zinc-200 px-4 py-3">
              <Network size={16} />
              <h2 className="text-sm font-semibold">Scoring Weights</h2>
            </div>
            <div className="space-y-3 p-4">
              {weights.map(([key, label, value]) => (
                <div key={key} className="grid grid-cols-[110px_1fr_36px] items-center gap-3 text-xs">
                  <span className="font-medium text-zinc-600">{label}</span>
                  <div className="h-2 rounded bg-zinc-100">
                    <div className="h-2 rounded bg-zinc-900" style={{ width: `${value}%` }} />
                  </div>
                  <span className="text-right font-semibold text-zinc-900">{value}%</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="space-y-5">
          {result ? (
            <>
              <div className="grid gap-5 xl:grid-cols-[1fr_340px]">
                <div className="rounded border border-zinc-200 bg-white">
                  <div className="flex items-center gap-2 border-b border-zinc-200 px-4 py-3">
                    <Sparkles size={16} />
                    <h2 className="text-sm font-semibold">Hiring Blueprint</h2>
                  </div>
                  <div className="grid gap-4 p-4 md:grid-cols-3">
                    <div>
                      <p className="text-xs font-medium text-zinc-500">Role</p>
                      <p className="mt-1 text-sm font-semibold">{result.blueprint.seniority} {result.blueprint.role_type}</p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-zinc-500">Experience</p>
                      <p className="mt-1 text-sm font-semibold">{result.blueprint.experience}</p>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-zinc-500">Industry</p>
                      <p className="mt-1 text-sm font-semibold">{result.blueprint.industry}</p>
                    </div>
                    <div className="md:col-span-3">
                      <p className="text-xs font-medium text-zinc-500">Required Skills</p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {result.blueprint.required_skills.map((skill) => <Pill key={skill}>{skill}</Pill>)}
                      </div>
                    </div>
                    <div className="md:col-span-3">
                      <p className="text-xs font-medium text-zinc-500">Inferred Concepts</p>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {result.blueprint.inferred_concepts.slice(0, 10).map((skill) => <Pill key={skill}>{skill}</Pill>)}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="rounded border border-zinc-200 bg-white">
                  <div className="flex items-center gap-2 border-b border-zinc-200 px-4 py-3">
                    <MessageSquareText size={16} />
                    <h2 className="text-sm font-semibold">Copilot</h2>
                  </div>
                  <div className="space-y-3 p-4">
                    <input
                      value={question}
                      onChange={(event) => setQuestion(event.target.value)}
                      className="w-full rounded border border-zinc-300 px-3 py-2 text-sm outline-none focus:border-zinc-950"
                    />
                    <button
                      onClick={askCopilot}
                      className="inline-flex w-full items-center justify-center gap-2 rounded border border-zinc-300 bg-white px-3 py-2 text-sm font-semibold hover:bg-zinc-50"
                    >
                      <MessageSquareText size={15} />
                      Ask
                    </button>
                    {copilotAnswer ? <p className="rounded border border-zinc-200 bg-zinc-50 p-3 text-sm leading-6 text-zinc-700">{copilotAnswer}</p> : null}
                  </div>
                </div>
              </div>

              <div className="grid gap-5 xl:grid-cols-[430px_1fr]">
                <div className="rounded border border-zinc-200 bg-white">
                  <div className="flex items-center gap-2 border-b border-zinc-200 px-4 py-3">
                    <Trophy size={16} />
                    <h2 className="text-sm font-semibold">Top Ranked Candidates</h2>
                  </div>
                  <div className="divide-y divide-zinc-100">
                    {result.matches.map((match, index) => (
                      <button
                        key={match.candidate.id}
                        onClick={() => setSelectedId(match.candidate.id)}
                        className={`grid w-full grid-cols-[42px_1fr_58px] items-center gap-3 px-4 py-3 text-left hover:bg-zinc-50 ${
                          selected?.candidate.id === match.candidate.id ? "bg-zinc-50" : ""
                        }`}
                      >
                        <span className="flex h-8 w-8 items-center justify-center rounded bg-zinc-100 text-xs font-bold text-zinc-700">
                          {index + 1}
                        </span>
                        <span className="min-w-0">
                          <span className="block truncate text-sm font-semibold">{match.candidate.name}</span>
                          <span className="block truncate text-xs text-zinc-500">{match.candidate.headline}</span>
                        </span>
                        <span className={`rounded px-2 py-1 text-center text-sm font-bold text-white ${scoreColor(match.score.final_score)}`}>
                          {Math.round(match.score.final_score)}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>

                {selected ? (
                  <div className="rounded border border-zinc-200 bg-white">
                    <div className="flex items-start justify-between gap-4 border-b border-zinc-200 px-4 py-4">
                      <div>
                        <h2 className="text-lg font-semibold">{selected.candidate.name}</h2>
                        <p className="text-sm text-zinc-500">{selected.candidate.headline} · {selected.candidate.location}</p>
                      </div>
                      <div className={`rounded px-3 py-2 text-lg font-bold text-white ${scoreColor(selected.score.final_score)}`}>
                        {Math.round(selected.score.final_score)}
                      </div>
                    </div>
                    <div className="space-y-5 p-4">
                      <div className="grid gap-3 md:grid-cols-3">
                        <Metric label="Semantic" value={selected.score.semantic_fit} icon={<Search size={14} />} />
                        <Metric label="Skills" value={selected.score.skill_match} icon={<BrainCircuit size={14} />} />
                        <Metric label="Experience" value={selected.score.experience_match} icon={<Users size={14} />} />
                        <Metric label="Activity" value={selected.score.activity_signals} icon={<Activity size={14} />} />
                        <Metric label="Leadership" value={selected.score.leadership} icon={<GitBranch size={14} />} />
                        <Metric label="Culture" value={selected.score.culture_fit} icon={<Sparkles size={14} />} />
                      </div>

                      <div>
                        <p className="text-xs font-medium uppercase text-zinc-500">Reason</p>
                        <p className="mt-2 text-sm leading-6 text-zinc-700">{selected.reason}</p>
                      </div>

                      <div className="grid gap-4 md:grid-cols-2">
                        <div>
                          <p className="text-xs font-medium uppercase text-zinc-500">Strengths</p>
                          <ul className="mt-2 space-y-2">
                            {selected.strengths.map((item) => (
                              <li key={item} className="flex gap-2 text-sm leading-6 text-zinc-700">
                                <ChevronRight className="mt-1 shrink-0 text-emerald-600" size={14} />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                        <div>
                          <p className="text-xs font-medium uppercase text-zinc-500">Weaknesses</p>
                          <ul className="mt-2 space-y-2">
                            {selected.weaknesses.map((item) => (
                              <li key={item} className="flex gap-2 text-sm leading-6 text-zinc-700">
                                <ChevronRight className="mt-1 shrink-0 text-amber-600" size={14} />
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>

                      <div>
                        <p className="text-xs font-medium uppercase text-zinc-500">Knowledge Graph Skills</p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {selected.candidate.expanded_skills.slice(0, 18).map((skill) => <Pill key={skill}>{skill}</Pill>)}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            </>
          ) : (
            <div className="flex min-h-[640px] items-center justify-center rounded border border-dashed border-zinc-300 bg-white">
              <div className="max-w-sm text-center">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded bg-zinc-950 text-white">
                  <Search size={22} />
                </div>
                <h2 className="mt-4 text-lg font-semibold">Run a candidate match</h2>
                <p className="mt-2 text-sm leading-6 text-zinc-500">
                  Paste a job description, generate the hiring blueprint, retrieve candidates, and inspect ranked explanations.
                </p>
              </div>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
