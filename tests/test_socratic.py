"""Tests for SocraticCreator using a stubbed language model (no network)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from open_notebook_creator_sdk import ContentBundle, CreationRequest, ModelRole
from open_notebook_creator_sdk.testing import (
    assert_creator_compliant,
    assert_result_compliant,
)
from socratic_creator import SocraticCreator


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, payload: str):
        self._payload = payload

    async def ainvoke(self, _prompt):
        return _FakeResp(self._payload)


class _QueueRole(ModelRole):
    """A ModelRole that returns canned payloads in order: first create_language
    call gets payloads[0] (the plan), subsequent calls cycle the rest (nuggets)."""

    payloads: list = []
    calls: int = 0

    def create_language(self, **_):
        i = min(self.calls, len(self.payloads) - 1)
        self.calls += 1
        return _FakeLLM(self.payloads[i])


def _plan(nuggets):
    return json.dumps({"title": "Test Session", "nuggets": nuggets})


def _nugget_payload(**over):
    d = {
        "question": "What does the mitochondria do?",
        "question_type": "probing",
        "expected_points": ["Produces ATP", "Site of cellular respiration"],
        "misconceptions": [
            {"label": "it stores DNA only", "question": "Then where does ATP come from?"}
        ],
        "hints": [
            "What molecule powers most cell processes?",
            "Which organelle is called the powerhouse?",
            "If respiration happens somewhere, where?",
        ],
        "reveal": {
            "answer": "It produces ATP through cellular respiration.",
            "citations": ["source:bio1"],
        },
        "deeper": "How would a cell behave without it?",
        "source_ids": ["source:bio1"],
    }
    d.update(over)
    return json.dumps(d)


def _request(td, payloads, config=None, sources=None):
    return CreationRequest(
        content=ContentBundle(
            text="Mitochondria produce ATP via cellular respiration.",
            sources=sources
            if sources is not None
            else [{"id": "source:bio1", "title": "Cell Biology"}],
        ),
        config=config or {"num_nuggets": 3},
        models={"text": _QueueRole(provider="fake", model="fake", payloads=payloads)},
        output_dir=td,
        artifact_id="art-1",
    )


def test_static_compliance():
    assert_creator_compliant(SocraticCreator())


@pytest.mark.asyncio
async def test_generate_produces_session_and_study_sheet():
    creator = SocraticCreator()
    plan = _plan(
        [
            {"title": "ATP production", "kind": "concept", "source_ids": ["source:bio1"], "summary": "s"},
            {"title": "Respiration", "kind": "concept", "source_ids": ["source:bio1"], "summary": "s"},
        ]
    )
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(_request(td, [plan, _nugget_payload()]))
        assert result.status == "SUCCESS"
        assert_result_compliant(creator, result)
        assert result.data["title"] == "Test Session"
        assert len(result.data["nuggets"]) == 2
        n = result.data["nuggets"][0]
        assert n["question"]
        assert len(n["hints"]) == 3
        assert n["reveal"]["citations"] == ["source:bio1"]
        assert result.data["allow_reveal"] is True
        # study sheet emitted, contained, questions-first with answers section
        assert len(result.files) == 1
        sheet = Path(td) / result.files[0].path
        assert sheet.exists()
        text = sheet.read_text()
        assert "## Answers" in text
        assert text.index("What does the mitochondria do?") < text.index("## Answers")


@pytest.mark.asyncio
async def test_synthesis_nuggets_require_all_concepts():
    creator = SocraticCreator()
    plan = _plan(
        [
            {"title": "A", "kind": "concept", "source_ids": [], "summary": "s"},
            {"title": "B", "kind": "concept", "source_ids": [], "summary": "s"},
            {"title": "Capstone", "kind": "synthesis", "source_ids": [], "summary": "s"},
        ]
    )
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(
            _request(td, [plan, _nugget_payload()], config={"num_nuggets": 3, "difficulty": "synthesis"})
        )
        assert result.status == "SUCCESS"
        by_kind = {n["kind"]: n for n in result.data["nuggets"]}
        assert set(by_kind["synthesis"]["requires"]) == {"n1", "n2"}
        assert by_kind["concept"]["requires"] == []


@pytest.mark.asyncio
async def test_invalid_plan_is_failure():
    creator = SocraticCreator()
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(_request(td, ["not json at all"]))
        assert result.status == "FAILURE"
        assert result.errors[0].phase == "plan"


@pytest.mark.asyncio
async def test_bad_nuggets_dropped_with_warning():
    creator = SocraticCreator()
    plan = _plan(
        [
            {"title": "Good", "kind": "concept", "source_ids": [], "summary": "s"},
            {"title": "Bad", "kind": "concept", "source_ids": [], "summary": "s"},
        ]
    )
    ok = _nugget_payload()
    bad = json.dumps({"question": "", "expected_points": [], "reveal": {"answer": ""}})
    with tempfile.TemporaryDirectory() as td:
        # plan, then first nugget ok, second nugget bad
        result = await creator.generate(_request(td, [plan, ok, bad]))
        assert result.status == "SUCCESS"
        assert len(result.data["nuggets"]) == 1
        assert any("skipped" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_all_nuggets_bad_is_failure():
    creator = SocraticCreator()
    plan = _plan([{"title": "Only", "kind": "concept", "source_ids": [], "summary": "s"}])
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(_request(td, [plan, "not json"]))
        assert result.status == "FAILURE"
        assert result.errors[0].phase == "generate"


@pytest.mark.asyncio
async def test_made_up_citations_are_filtered():
    creator = SocraticCreator()
    plan = _plan(
        [{"title": "T", "kind": "concept", "source_ids": ["source:bio1"], "summary": "s"}]
    )
    payload = _nugget_payload(
        reveal={"answer": "Grounded answer.", "citations": ["source:bio1", "source:FAKE"]}
    )
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(_request(td, [plan, payload]))
        assert result.status == "SUCCESS"
        assert result.data["nuggets"][0]["reveal"]["citations"] == ["source:bio1"]


@pytest.mark.asyncio
async def test_no_text_role_is_failure():
    creator = SocraticCreator()
    with tempfile.TemporaryDirectory() as td:
        req = CreationRequest(
            content=ContentBundle(text="x"), output_dir=td, artifact_id="a"
        )
        result = await creator.generate(req)
        assert result.status == "FAILURE"
        assert result.errors[0].phase == "setup"


@pytest.mark.asyncio
async def test_allow_reveal_false_rides_on_data():
    creator = SocraticCreator()
    plan = _plan([{"title": "T", "kind": "concept", "source_ids": [], "summary": "s"}])
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(
            _request(td, [plan, _nugget_payload()], config={"allow_reveal": False})
        )
        assert result.status == "SUCCESS"
        assert result.data["allow_reveal"] is False


def test_manifest_declares_view_bundle_and_it_ships():
    """The creator owns its UI: the manifest points at a shipped HTML view bundle."""
    from importlib import resources

    m = SocraticCreator().manifest
    assert m.view is not None
    assert m.view.entry == "view/index.html"
    asset = resources.files("socratic_creator").joinpath(m.view.entry)
    assert asset.is_file()
    html = asset.read_text()
    # self-contained + speaks the host handshake + dispatches our schema
    assert "open-notebook:ready" in html
    assert "open-notebook:artifact" in html
    assert "socratic.v1" in html
    assert "<script src" not in html  # no external scripts (sandbox-safe, offline)


def test_view_bundle_never_leaks_the_reveal_early():
    """The structural guarantee: the reveal renders only from the checklist
    screen ('Show the model answer') or the exhausted-hints path — the answer
    text is never injected while the opening question is on screen."""
    from importlib import resources

    html = resources.files("socratic_creator").joinpath("view/index.html").read_text()
    # hint ladder uses hints only; reveal.answer appears solely in the
    # checklist-screen reveal handler
    assert html.count("reveal.answer") == 1
    assert "Show the model answer" in html
