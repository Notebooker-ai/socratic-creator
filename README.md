# socratic-creator

An [Open Notebook](https://github.com/Notebooker-ai/open-notebook-nb) creator that
turns a notebook's sources into a **precomputed Socratic tutoring session**
(`socratic.v1`), plus a printable markdown study sheet.

## Why precomputed

Prompt-only Socratic tutors have one famous failure mode: the model caves and
blurts the answer after a couple of follow-ups. This creator generates the whole
dialogue tree up front — opening question, self-assessment checklist,
misconception counter-questions, an escalating ladder of *sub-question* hints,
and a citation-pinned reveal per nugget. The shipped view bundle then runs the
session entirely client-side with **no runtime LLM**, so the tutor structurally
cannot reveal early: the answer simply is not on screen until the flow reaches it.

## What it generates

- A **planner pass** extracts 3–25 teachable nuggets, ordered foundational →
  nuanced, each pinned to the source ids that teach it. An optional `scope`
  config ("only material through chapter 5") excludes later content as spoilers.
- A **per-nugget pass** (bounded fan-out) writes, for each nugget:
  - one opening question, typed `clarifying | probing | connecting | counter |
    hypothetical`
  - 2–6 `expected_points` — the self-assessment checklist
  - up to 3 misconception probes: a wrong-belief label + the counter-question
    that exposes the contradiction (never a correction)
  - exactly 3 escalating hints, each a smaller *question*, not a statement
  - a reveal grounded only in the sources, with citations
  - an optional `deeper` stretch question, unlocked by full mastery
- At `difficulty: synthesis`, the final nuggets are capstones gated behind all
  concept nuggets (`requires`).

## Config

| field | default | notes |
| --- | --- | --- |
| `num_nuggets` | 10 | 3–25 |
| `difficulty` | `application` | `recall` / `application` / `synthesis` |
| `persona` | "a patient, curious professor" | the tutoring voice |
| `allow_reveal` | `true` | off = pure Socratic; the answer is never shown |
| `scope` | — | spoiler boundary, free text |

## Development

```bash
uv sync --extra dev
uv run pytest
```

Tests stub the language model — no network, no keys. `assert_creator_compliant`
from the SDK's compliance suite runs in CI.
