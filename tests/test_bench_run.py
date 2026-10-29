import json
from pathlib import Path

import pytest
from tests.conftest import make_settings
from tests.fakes import install

from verifai.bench.cli import main
from verifai.bench.datasets import Example, LoadedDataset
from verifai.bench.labels import BenchLabel
from verifai.bench.report import RunMeta, git_sha, write_report
from verifai.bench.scoring import evaluate, two_source_blocked
from verifai.config import Settings
from verifai.corroboration import Stance
from verifai.experts.citation import build_citation_expert
from verifai.experts.health import HealthExpert
from verifai.experts.router import Router
from verifai.http import PoliteHttp
from verifai.models.judge import NullJudge
from verifai.orchestrate import run_check
from verifai.retrieval.types import Passage, RetrievalResult
from verifai.schemas import Verdict

CLAIM = "A clinical trial found this drug lowers blood pressure in patients with diabetes."


def _dataset(*rows: tuple[str, str, str]) -> LoadedDataset:
    examples = tuple(
        Example(
            id=item_id,
            dataset="scifact",
            domain="health",
            text=text,
            gold=BenchLabel(gold),
            raw_label=gold,
        )
        for item_id, text, gold in rows
    )
    return LoadedDataset(
        name="scifact",
        domain="health",
        version="test",
        source_url="fixture://local",
        sha256="0" * 64,
        split="fixture",
        license_note="Test fixture.",
        mapping_notes=("fixture",),
        examples=examples,
        skipped=0,
    )


def _passage(pmid: str, stance: Stance) -> Passage:
    return Passage(
        source="pubmed",
        family="pubmed",
        title=f"Trial {pmid}",
        url=None,
        text="The trial reported a blood pressure result in patients.",
        document_ids=(f"pmid:{pmid}",),
        native_stance=stance,
        native_confidence=0.9,
    )


class MapRetriever:
    name = "fixture"

    def __init__(self, rows: dict[str, list[Passage]]) -> None:
        self.rows = rows

    def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(self.name, passages=list(self.rows.get(query, [])))


def test_two_papers_support_and_one_paper_is_blocked() -> None:
    supported = CLAIM
    blocked = "A second clinical trial found the same drug raises blood pressure in patients."
    empty = "A third clinical trial has no retrieved studies about this drug at all."
    retriever = MapRetriever(
        {
            supported: [_passage("1", Stance.SUPPORTS), _passage("2", Stance.SUPPORTS)],
            blocked: [_passage("3", Stance.SUPPORTS)],
            empty: [],
        }
    )
    expert = HealthExpert([retriever], NullJudge())
    router = Router([expert])
    dataset = _dataset(
        ("s", supported, "supported"),
        ("b", blocked, "unverified"),
        ("e", empty, "unverified"),
    )
    result = evaluate(router, [dataset], seed=0, limit=None)
    by_id = {row.example_id: row for row in result.datasets[0].outcomes}
    assert by_id["s"].predicted == BenchLabel.SUPPORTED
    assert by_id["s"].blocked is False
    assert by_id["s"].experts == ("health",)
    assert by_id["b"].predicted == BenchLabel.UNVERIFIED
    assert by_id["b"].blocked is True
    assert by_id["e"].predicted == BenchLabel.UNVERIFIED
    assert by_id["e"].blocked is False
    assert result.by_domain["health"].two_source_blocks == 1


def test_mixed_verdict_is_unverified_and_not_a_block() -> None:
    retriever = MapRetriever(
        {
            CLAIM: [
                _passage("1", Stance.SUPPORTS),
                _passage("2", Stance.SUPPORTS),
                _passage("3", Stance.REFUTES),
                _passage("4", Stance.REFUTES),
            ]
        }
    )
    router = Router([HealthExpert([retriever], NullJudge())])
    bundle = run_check(router, CLAIM, limit=1)
    assert bundle.verdict is Verdict.MIXED
    assert two_source_blocked(bundle) is False
    result = evaluate(router, [_dataset(("m", CLAIM, "unverified"))], seed=1, limit=0)
    assert result.datasets[0].outcomes[0].predicted == BenchLabel.UNVERIFIED
    assert result.datasets[0].outcomes[0].blocked is False


def test_citation_rows_use_the_citation_expert(
    settings: Settings, http_mock: object, polite_http: PoliteHttp
) -> None:
    install(http_mock)
    expert = build_citation_expert(polite_http, settings)
    router = Router([expert])
    dataset = LoadedDataset(
        name="citations",
        domain="citations",
        version="handwritten-1",
        source_url="fixture",
        sha256="1" * 64,
        split="handwritten",
        license_note="test",
        mapping_notes=("test",),
        examples=(
            Example(
                id="attention-doi",
                dataset="citations",
                domain="citations",
                text="https://doi.org/10.48550/arXiv.1706.03762",
                gold=BenchLabel.SUPPORTED,
                raw_label="supported",
            ),
            Example(
                id="fake-doi-1234",
                dataset="citations",
                domain="citations",
                text="10.1234/verifai.fabricated.2021",
                gold=BenchLabel.REFUTED,
                raw_label="refuted",
            ),
        ),
        skipped=0,
    )
    result = evaluate(router, [dataset], seed=0, limit=None)
    by_id = {row.example_id: row for row in result.datasets[0].outcomes}
    assert by_id["attention-doi"].predicted == BenchLabel.SUPPORTED
    assert by_id["attention-doi"].blocked is False
    assert by_id["fake-doi-1234"].predicted == BenchLabel.REFUTED
    assert "citation" in by_id["fake-doi-1234"].experts


def test_cli_offline_uses_recorded_passages_and_the_real_rule(tmp_path: Path) -> None:
    claim = CLAIM
    fever = tmp_path / "fever.jsonl"
    fever.write_text(
        json.dumps({"id": 1, "label": "SUPPORTS", "claim": claim}) + "\n",
        encoding="utf-8",
    )
    fixtures = tmp_path / "passages.json"
    fixtures.write_text(
        json.dumps(
            {
                claim: [
                    {
                        "source": "pubmed",
                        "family": "pubmed",
                        "title": "Trial one",
                        "text": "Blood pressure fell.",
                        "document_ids": ["pmid:1"],
                        "native_stance": "supports",
                        "native_confidence": 0.9,
                    },
                    {
                        "source": "europepmc",
                        "family": "europepmc",
                        "title": "Trial two",
                        "text": "Blood pressure fell again.",
                        "document_ids": ["pmid:2"],
                        "native_stance": "supports",
                        "native_confidence": 0.8,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "--datasets",
            "fever",
            "--backend",
            "none",
            "--offline",
            "--fever",
            str(fever),
            "--passage-fixtures",
            str(fixtures),
            "--limit",
            "5",
            "--seed",
            "3",
            "--json-out",
            str(tmp_path / "out.json"),
            "--markdown-out",
            str(tmp_path / "out.md"),
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["backend"] == "none"
    assert payload["model_id"] == "none"
    assert payload["seed"] == 3
    assert payload["offline"] is True
    assert payload["examples"][0]["predicted"] == "supported"
    assert payload["examples"][0]["blocked_by_two_source_rule"] is False
    assert payload["examples"][0]["experts"] == ["health"]
    assert main(["--datasets", "citations", "--offline"]) == 1
    assert main(["--datasets", "nope"]) == 1


def test_report_records_sha_seed_and_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset(("s", CLAIM, "unverified"))
    router = Router([HealthExpert([MapRetriever({CLAIM: []})], NullJudge())])
    result = evaluate(router, [dataset], seed=4, limit=10)
    meta = RunMeta(
        date="2026-10-29T00:00:00Z",
        git_sha="abc123",
        backend="none",
        model_id="none",
        judge_name="none",
        seed=4,
        limit=10,
        offline=True,
    )
    json_path = tmp_path / "r.json"
    markdown_path = tmp_path / "r.md"
    write_report(result, meta, json_path=json_path, markdown_path=markdown_path)
    body = json.loads(json_path.read_text(encoding="utf-8"))
    assert body["git_sha"] == "abc123"
    assert body["seed"] == 4
    assert body["model_id"] == "none"
    assert "abc123" in markdown_path.read_text(encoding="utf-8")
    monkeypatch.setattr("verifai.bench.report.shutil.which", lambda _name: None)
    assert git_sha(tmp_path) == "unknown"
    assert make_settings().model_backend == "none"
