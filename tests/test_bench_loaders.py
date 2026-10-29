import io
import json
import tarfile
import zipfile
from pathlib import Path

import httpx
import pytest

from verifai.bench.datasets import (
    download,
    load_citations,
    load_fever,
    load_liar,
    load_scifact_claims,
    read_tar_member,
)
from verifai.bench.labels import BenchLabel, scifact_gold
from verifai.experts.citation.extract import extract_citations


def test_fever_and_scifact_maps(tmp_path: Path) -> None:
    fever = tmp_path / "fever.jsonl"
    fever.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"id": 7, "label": "SUPPORTS", "claim": "Cats are mammals and have claws."},
                {"id": 8, "label": "REFUTES", "claim": "The moon is a star made of cheese."},
                {"id": 9, "label": "NOT ENOUGH INFO", "claim": "A secret number is even today."},
                {"id": 10, "label": "MAYBE", "claim": "This label is not in FEVER at all."},
            )
        ),
        encoding="utf-8",
    )
    loaded = load_fever(fever)
    assert [item.gold for item in loaded.examples] == [
        BenchLabel.SUPPORTED,
        BenchLabel.REFUTED,
        BenchLabel.UNVERIFIED,
    ]
    assert loaded.skipped == 1
    claims = tmp_path / "claims.jsonl"
    claims.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {
                    "id": 1,
                    "claim": "Drug A lowers blood pressure.",
                    "evidence": {"10": [{"label": "SUPPORT"}]},
                },
                {
                    "id": 2,
                    "claim": "Drug A raises blood pressure.",
                    "evidence": {"11": [{"label": "CONTRADICT"}]},
                },
                {"id": 3, "claim": "Drug A tastes like mint.", "evidence": {}},
                {
                    "id": 4,
                    "claim": "Drug A both helps and harms.",
                    "evidence": {"12": [{"label": "SUPPORT"}], "13": [{"label": "CONTRADICT"}]},
                },
            )
        ),
        encoding="utf-8",
    )
    scifact = load_scifact_claims(claims, corpus_sha256="abc")
    assert scifact.extra_sha256 == "abc"
    assert [item.gold for item in scifact.examples] == [
        BenchLabel.SUPPORTED,
        BenchLabel.REFUTED,
        BenchLabel.UNVERIFIED,
        BenchLabel.UNVERIFIED,
    ]
    assert scifact_gold({"1": "nope"}) == BenchLabel.UNVERIFIED


def test_liar_six_way_mapping(tmp_path: Path) -> None:
    path = tmp_path / "test.tsv"
    rows = [
        ["1", "true", "Statement one."],
        ["2", "mostly-true", "Statement two."],
        ["3", "false", "Statement three."],
        ["4", "pants-fire", "Statement four."],
        ["5", "half-true", "Statement five."],
        ["6", "barely-true", "Statement six."],
        ["7", "mostly true", "A spaced label is not the dataset spelling."],
    ]
    path.write_text("".join("\t".join(row) + "\n" for row in rows), encoding="utf-8")
    loaded = load_liar(path)
    assert [item.gold.value for item in loaded.examples] == [
        "supported",
        "supported",
        "refuted",
        "refuted",
        "unverified",
        "unverified",
    ]
    assert loaded.skipped == 1
    assert "research use only" in loaded.license_note.lower()


def test_handwritten_citations_extract_once() -> None:
    loaded = load_citations()
    assert loaded.version == "handwritten-1"
    assert len(loaded.examples) == 16
    for example in loaded.examples:
        assert len(extract_citations(example.text)) == 1


def test_bad_citation_label_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"id": "x", "label": "maybe", "text": "https://doi.org/10.1000/abc"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="maybe"):
        load_citations(path)


def test_download_and_archives(tmp_path: Path) -> None:
    dest = tmp_path / "file.jsonl"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{}\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    download("https://example.test/file.jsonl", dest, client=client)
    download("https://example.test/file.jsonl", dest, client=client)
    assert dest.read_bytes() == b"{}\n"
    client.close()

    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as archive:
        payload = b"claim\n"
        info = tarfile.TarInfo(name="data/claims_dev.jsonl")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    assert read_tar_member(blob.getvalue(), "claims_dev.jsonl") == payload

    archive_path = tmp_path / "liar.zip"
    table = tmp_path / "test.tsv"
    with zipfile.ZipFile(archive_path, "w") as bundle:
        bundle.writestr("liar_dataset/test.tsv", "9\ttrue\tA real statement here.\n")
    from verifai.bench.datasets import _extract_liar_test

    _extract_liar_test(archive_path, table)
    assert load_liar(table).examples[0].gold == BenchLabel.SUPPORTED
