"""socratic-creator: an Open Notebook creator that turns notebook content into a
precomputed **Socratic tutoring session** (emitted as ``socratic.v1``) plus a
printable markdown study sheet.

The whole dialogue tree is generated at creation time — a planner pass extracts
nuggets pinned to specific sources, then a per-nugget pass writes the question,
a self-assessment checklist, misconception counter-questions, an escalating
hint ladder (sub-questions, never statements), and a citation-pinned reveal.
The shipped view bundle runs the tutoring loop entirely client-side with no
runtime LLM, so the tutor structurally cannot cave and answer early.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional

from ai_prompter import Prompter
from loguru import logger
from open_notebook_creator_sdk import (
    BaseCreator,
    CreationError,
    CreationFile,
    CreationRequest,
    CreationResult,
    CreatorManifest,
    CreatorView,
    ModelRoleSpec,
)
from open_notebook_creator_sdk.schemas.socratic_v1 import (
    SocraticNugget,
    SocraticProbe,
    SocraticReveal,
    SocraticV1,
)
from pydantic import BaseModel, Field

__version__ = "0.1.0"

SCHEMA_ID = "socratic.v1"
_MAX_CONCURRENT_NUGGETS = 4
_QUESTION_TYPES = {"clarifying", "probing", "connecting", "counter", "hypothetical"}


class SocraticConfig(BaseModel):
    """Per-generation config; drives the host's generate form."""

    num_nuggets: int = Field(
        default=10, ge=3, le=25, description="How many concepts to tutor through"
    )
    difficulty: Literal["recall", "application", "synthesis"] = Field(
        default="application",
        description=(
            "recall = definitions and facts; application = reasoning with the "
            "material; synthesis = adds capstone questions that combine concepts"
        ),
    )
    persona: str = Field(
        default="a patient, curious professor",
        description="The tutoring voice (e.g. 'a drill sergeant', 'a friendly peer')",
    )
    allow_reveal: bool = Field(
        default=True,
        description=(
            "Show the model answer once every hint is exhausted. Off = pure "
            "Socratic: the session never states the answer."
        ),
    )
    scope: str = Field(
        default="",
        description=(
            "Optional boundary, e.g. 'only material through chapter 5' — later "
            "content is treated as spoilers and excluded"
        ),
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _read_prompt(name: str) -> str:
    return resources.files("socratic_creator.prompts").joinpath(name).read_text()


def _parse_json(raw: str) -> Optional[Any]:
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None


def _clean_str_list(value: Any, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    out = [str(v).strip() for v in value if str(v).strip()]
    return out[:limit]


def _valid_source_ids(value: Any, known: set) -> List[str]:
    """Keep only source ids that exist in the bundle's provenance (no made-up
    citations); pass everything through when the host sent no provenance."""
    ids = _clean_str_list(value, 8)
    if not known:
        return ids
    return [i for i in ids if i in known]


class SocraticCreator(BaseCreator):
    config_model: ClassVar[type] = SocraticConfig

    @property
    def manifest(self) -> CreatorManifest:
        return self.build_manifest(
            key="socratic",
            name="Socratic Tutor",
            version=__version__,
            description=(
                "Guided-discovery tutoring: questions, hints, and misconception "
                "probes precomputed from your sources — the tutor never blurts "
                "the answer."
            ),
            sdk_compat=">=0.6,<1",
            emits=[SCHEMA_ID],
            model_roles=[
                ModelRoleSpec(
                    key="text",
                    kind="language",
                    requires=["structured_json"],
                    description="LLM that plans the curriculum and writes each dialogue nugget.",
                )
            ],
            icon="messages-square",
            view=CreatorView(entry="view/index.html"),
        )

    async def generate(self, request: CreationRequest) -> CreationResult:
        cfg = SocraticConfig.model_validate(request.config)
        role = request.models.get("text")
        if role is None:
            return CreationResult(
                status="FAILURE",
                schema_id=SCHEMA_ID,
                data={},
                errors=[CreationError(phase="setup", message="missing 'text' model role")],
                user_message="No language model was provided for the Socratic tutor.",
            )

        known_sources = {
            str(s.get("id"))
            for s in (request.content.sources or [])
            if isinstance(s, dict) and s.get("id")
        }

        # ---- phase 1: plan the curriculum ---------------------------------
        plan_prompt = Prompter(template_text=_read_prompt("plan.jinja")).render(
            {
                "content": request.content.text,
                "sources": request.content.sources,
                "num_nuggets": cfg.num_nuggets,
                "difficulty": cfg.difficulty,
                "scope": cfg.scope,
                "language": request.language,
                "instructions": request.instructions,
            }
        )
        llm = role.create_language(structured={"type": "json"}, max_tokens=4000)
        resp = await llm.ainvoke(plan_prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        plan = _parse_json(raw)
        planned = plan.get("nuggets") if isinstance(plan, dict) else None
        if not isinstance(planned, list) or not planned:
            logger.error("socratic: planner returned no usable curriculum")
            return CreationResult(
                status="FAILURE",
                schema_id=SCHEMA_ID,
                data={},
                errors=[
                    CreationError(
                        phase="plan",
                        message="planner returned no usable curriculum",
                        retryable=True,
                    )
                ],
                user_message=(
                    "The model could not plan a tutoring session from this "
                    "content. Please retry."
                ),
            )

        planned = planned[: cfg.num_nuggets]
        plan_items: List[Dict[str, Any]] = []
        for item in planned:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            plan_items.append(
                {
                    "id": f"n{len(plan_items) + 1}",
                    "title": title,
                    "kind": "synthesis" if item.get("kind") == "synthesis" else "concept",
                    "source_ids": _valid_source_ids(item.get("source_ids"), known_sources),
                    "summary": str(item.get("summary") or "").strip(),
                }
            )
        if not plan_items:
            return CreationResult(
                status="FAILURE",
                schema_id=SCHEMA_ID,
                data={},
                errors=[CreationError(phase="plan", message="no valid nuggets in plan")],
                user_message="No tutoring topics could be planned from this content.",
            )

        # Synthesis nuggets require every prior concept nugget.
        concept_ids = [p["id"] for p in plan_items if p["kind"] == "concept"]
        for p in plan_items:
            p["requires"] = concept_ids.copy() if p["kind"] == "synthesis" else []

        # ---- phase 2: write each nugget's dialogue (bounded fan-out) ------
        nugget_template = _read_prompt("nugget.jinja")
        sem = asyncio.Semaphore(_MAX_CONCURRENT_NUGGETS)

        async def write_nugget(p: Dict[str, Any]) -> Optional[SocraticNugget]:
            prompt = Prompter(template_text=nugget_template).render(
                {
                    "content": request.content.text,
                    "nugget": p,
                    "persona": cfg.persona,
                    "difficulty": cfg.difficulty,
                    "language": request.language,
                    "instructions": request.instructions,
                }
            )
            async with sem:
                try:
                    n_llm = role.create_language(
                        structured={"type": "json"}, max_tokens=2500
                    )
                    n_resp = await n_llm.ainvoke(prompt)
                except Exception as e:  # noqa: BLE001 - one nugget must not kill the run
                    logger.warning(f"socratic: nugget '{p['title']}' failed: {e}")
                    return None
            n_raw = n_resp.content if hasattr(n_resp, "content") else str(n_resp)
            d = _parse_json(n_raw)
            if not isinstance(d, dict):
                logger.warning(f"socratic: nugget '{p['title']}' returned non-JSON")
                return None
            question = str(d.get("question") or "").strip()
            answer = str((d.get("reveal") or {}).get("answer") or "").strip()
            expected = _clean_str_list(d.get("expected_points"), 6)
            hints = _clean_str_list(d.get("hints"), 3)
            if not question or not answer or not expected:
                logger.warning(f"socratic: nugget '{p['title']}' incomplete; dropped")
                return None
            qtype = str(d.get("question_type") or "probing")
            probes = []
            for m in d.get("misconceptions") or []:
                if isinstance(m, dict) and m.get("label") and m.get("question"):
                    probes.append(
                        SocraticProbe(
                            label=str(m["label"]).strip(),
                            question=str(m["question"]).strip(),
                        )
                    )
            deeper = str(d.get("deeper") or "").strip() or None
            return SocraticNugget(
                id=p["id"],
                title=p["title"],
                kind=p["kind"],
                requires=p["requires"],
                source_ids=_valid_source_ids(
                    d.get("source_ids") or p["source_ids"], known_sources
                ),
                question=question,
                question_type=qtype if qtype in _QUESTION_TYPES else "probing",
                expected_points=expected,
                hints=hints,
                misconceptions=probes[:3],
                reveal=SocraticReveal(
                    answer=answer,
                    citations=_valid_source_ids(
                        (d.get("reveal") or {}).get("citations"), known_sources
                    )
                    or p["source_ids"],
                ),
                deeper=deeper,
            )

        written = await asyncio.gather(*(write_nugget(p) for p in plan_items))
        nuggets = [n for n in written if n is not None]
        dropped = len(plan_items) - len(nuggets)
        if not nuggets:
            return CreationResult(
                status="FAILURE",
                schema_id=SCHEMA_ID,
                data={},
                errors=[
                    CreationError(
                        phase="generate",
                        message="no valid nuggets produced",
                        retryable=True,
                    )
                ],
                user_message="No tutoring dialogue could be generated. Please retry.",
            )
        # Drop dangling requires if a prerequisite nugget was dropped.
        kept = {n.id for n in nuggets}
        for n in nuggets:
            n.requires = [r for r in n.requires if r in kept]

        title = ""
        if isinstance(plan, dict):
            title = str(plan.get("title") or "").strip()
        # allow_reveal rides in the schema: the view keeps reveals hidden until
        # hints are exhausted, and never shows them at all when it's False.
        data = SocraticV1(
            title=title or "Socratic Session",
            persona=cfg.persona,
            difficulty=cfg.difficulty,
            allow_reveal=cfg.allow_reveal,
            nuggets=nuggets,
        ).model_dump()

        warnings: List[str] = []
        errors: List[CreationError] = []
        if dropped:
            warnings.append(
                f"{dropped} planned topic(s) could not be written and were skipped."
            )

        # ---- study-sheet export (best-effort -> PARTIAL on failure) -------
        files: List[CreationFile] = []
        rel_name = "socratic-study-sheet.md"
        try:
            out_path = Path(request.output_dir) / rel_name
            await asyncio.to_thread(
                out_path.write_text, _study_sheet_md(data), "utf-8"
            )
            files.append(
                CreationFile(
                    filename=rel_name,
                    content_type="text/markdown",
                    path=rel_name,
                    label="study_sheet",
                )
            )
        except Exception as e:  # noqa: BLE001 - export is non-fatal
            logger.warning(f"socratic: study sheet export failed: {e}")
            warnings.append("Study-sheet export failed; the session is still available.")
            errors.append(CreationError(phase="export", message=str(e)))

        return CreationResult(
            status="PARTIAL" if errors else "SUCCESS",
            schema_id=SCHEMA_ID,
            data=data,
            files=files,
            warnings=warnings,
            errors=errors,
        )


def _study_sheet_md(data: Dict[str, Any]) -> str:
    """A printable questions-first sheet: questions and hints up top, reveals at
    the back (like a good problem-set book), so it stays useful on paper."""
    lines = [f"# {data['title']}", ""]
    lines.append(
        f"_A Socratic study sheet · {data['difficulty']} level · answers at the back._"
    )
    lines.append("")
    for i, n in enumerate(data["nuggets"], 1):
        lines.append(f"## {i}. {n['title']}")
        lines.append("")
        lines.append(n["question"])
        lines.append("")
        if n["hints"]:
            lines.append("<details><summary>Hints</summary>")
            lines.append("")
            for j, h in enumerate(n["hints"], 1):
                lines.append(f"{j}. {h}")
            lines.append("")
            lines.append("</details>")
            lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    for i, n in enumerate(data["nuggets"], 1):
        lines.append(f"### {i}. {n['title']}")
        lines.append("")
        for p in n["expected_points"]:
            lines.append(f"- {p}")
        lines.append("")
        lines.append(n["reveal"]["answer"])
        if n["reveal"]["citations"]:
            lines.append("")
            lines.append(
                "Sources: " + ", ".join(f"[{c}]" for c in n["reveal"]["citations"])
            )
        lines.append("")
    return "\n".join(lines)
