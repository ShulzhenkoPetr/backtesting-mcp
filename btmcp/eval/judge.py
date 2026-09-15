from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from btmcp.eval.models import Model

RUBRIC_SYSTEM = (
    "You grade transcripts of an assistant using a backtesting tool server. "
    "Grade only the final answer's quality against the rubric, on a 1-5 scale. "
    "Do not reward or penalise tool choice; deterministic checks already cover that. "
    "1 = does not address the rubric, 3 = addresses it with gaps, 5 = fully satisfies it."
)


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=1, le=5)
    justification: str = Field(max_length=400)


def coerce_grade(payload: dict[str, Any]) -> dict[str, Any]:
    """Brings a provider's reply inside the bounds `Grade` declares.

    Strict dialects strip `minimum`, `maximum` and `maxLength` from the response
    schema because the provider refuses to accept them, so nothing server-side
    enforces the bounds the model here still validates against. Clamping is the
    honest reading of an out-of-range grade - a 7 means "top of the scale" - and it
    stops one verbose justification from ending a sweep several hours in.
    """
    out = dict(payload)
    score = out.get("score")
    if isinstance(score, int | float | str):
        with contextlib.suppress(TypeError, ValueError):
            out["score"] = max(1, min(5, int(float(score))))
    justification = out.get("justification")
    if isinstance(justification, str) and len(justification) > 400:
        out["justification"] = justification[:400]
    return out


class Judge:
    """Rubric grading with a transcript-keyed cache.

    Cached by transcript hash so re-running a report costs nothing, which matters
    because reports get regenerated far more often than runs do.
    """

    def __init__(self, model: Model, cache_path: Path | None = None) -> None:
        self.model = model
        self.model_id = model.id
        self.cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = {}
        if cache_path and cache_path.exists():
            self._cache = json.loads(cache_path.read_text())

    @staticmethod
    def key(rubric: str, final_answer: str, model_id: str) -> str:
        payload = json.dumps([rubric, final_answer, model_id], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def cached(self, rubric: str, final_answer: str) -> Grade | None:
        hit = self._cache.get(self.key(rubric, final_answer, self.model_id))
        return Grade.model_validate(hit) if hit else None

    def _store(self, rubric: str, final_answer: str, grade: Grade) -> None:
        self._cache[self.key(rubric, final_answer, self.model_id)] = grade.model_dump()
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))

    async def grade(self, rubric: str, final_answer: str) -> Grade:
        hit = self.cached(rubric, final_answer)
        if hit is not None:
            return hit

        payload = await self.model.complete_json(
            system=RUBRIC_SYSTEM,
            prompt=(
                f"Rubric:\n{rubric}\n\n"
                f"Assistant's final answer:\n{final_answer or '(no final answer was produced)'}"
            ),
            schema=Grade.model_json_schema(),
            name="rubric_grade",
        )
        grade = Grade.model_validate(coerce_grade(payload))
        self._store(rubric, final_answer, grade)
        return grade


class StubJudge:
    """Deterministic stand-in used by the test suite so grading needs no API key."""

    def __init__(self, score: int = 4) -> None:
        self.model_id = "stub"
        self.score = score
        self.calls = 0

    async def grade(self, rubric: str, final_answer: str) -> Grade:
        self.calls += 1
        score = 1 if not final_answer.strip() else self.score
        return Grade(score=score, justification="stub")
