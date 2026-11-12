"""Labeled datasets. Public files download into a cache and are not committed.

Splits:

- FEVER v1.0 labelled shared-task development set. The blind test labels
  were not public.
- SciFact v1.0 ``claims_dev.jsonl``. The test split has no gold labels.
- LIAR v1.0 ``test.tsv``, the public split papers usually report.
- A hand-written citation file that lives in this repository.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from verifai.bench.labels import (
    CITATION_MAPPING_NOTES,
    FEVER_LABELS,
    FEVER_MAPPING_NOTES,
    LIAR_LABELS,
    LIAR_MAPPING_NOTES,
    SCIFACT_MAPPING_NOTES,
    BenchLabel,
    bench_label,
    scifact_gold,
)

FEVER_URL = "https://fever.ai/download/fever/shared_task_dev.jsonl"
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
LIAR_URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
USER_AGENT = (
    "VerifAI/0.3.0 (https://github.com/Timeless-Dave/verifAI; mailto:verifai-dev@example.com)"
)

DOMAIN_GENERAL = "general"
DOMAIN_HEALTH = "health"
DOMAIN_NEWS = "news"
DOMAIN_CITATIONS = "citations"

DOMAIN_DISPLAY = {
    DOMAIN_GENERAL: "general",
    DOMAIN_HEALTH: "health/science",
    DOMAIN_NEWS: "news",
    DOMAIN_CITATIONS: "citations",
}


@dataclass(frozen=True)
class Example:
    id: str
    dataset: str
    domain: str
    text: str
    gold: BenchLabel
    raw_label: str


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    domain: str
    version: str
    source_url: str
    sha256: str
    split: str
    license_note: str
    mapping_notes: tuple[str, ...]
    examples: tuple[Example, ...]
    skipped: int
    extra_sha256: str | None = None


def citations_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "citations.jsonl"


def default_cache_dir() -> Path:
    return Path.cwd() / ".cache" / "verifai-bench"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    dest: Path,
    *,
    timeout: float = 120.0,
    client: httpx.Client | None = None,
) -> None:
    """Download ``url`` to ``dest``. An existing non-empty file is left in place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    partial = dest.with_suffix(dest.suffix + ".part")
    owns_client = client is None
    http = client or httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with http.stream("GET", url) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        partial.replace(dest)
    except (httpx.HTTPError, OSError):
        partial.unlink(missing_ok=True)
        raise
    finally:
        if owns_client:
            http.close()


def load_fever(path: Path) -> LoadedDataset:
    return _jsonl(
        path,
        name="fever",
        domain=DOMAIN_GENERAL,
        version="FEVER v1.0 shared_task_dev",
        source_url=FEVER_URL,
        split="shared_task_dev",
        license_note=(
            "FEVER v1.0 labelled development split from https://fever.ai/dataset/fever.html. "
            "Downloaded into the cache and not committed."
        ),
        mapping_notes=FEVER_MAPPING_NOTES,
        parse_row=_fever_row,
    )


def load_scifact_claims(path: Path, *, corpus_sha256: str | None = None) -> LoadedDataset:
    dataset = _jsonl(
        path,
        name="scifact",
        domain=DOMAIN_HEALTH,
        version="SciFact v1.0 claims_dev",
        source_url=SCIFACT_URL,
        split="claims_dev",
        license_note=(
            "AllenAI SciFact claims_dev.jsonl. Expert-written scientific claims. "
            "Downloaded into the cache and not committed. VerifAI does not read the "
            "SciFact corpus or the gold cited documents; the health expert searches "
            "PubMed and Europe PMC."
        ),
        mapping_notes=SCIFACT_MAPPING_NOTES,
        parse_row=_scifact_row,
    )
    if corpus_sha256 is None:
        return dataset
    return LoadedDataset(
        name=dataset.name,
        domain=dataset.domain,
        version=dataset.version,
        source_url=dataset.source_url,
        sha256=dataset.sha256,
        split=dataset.split,
        license_note=dataset.license_note,
        mapping_notes=dataset.mapping_notes,
        examples=dataset.examples,
        skipped=dataset.skipped,
        extra_sha256=corpus_sha256,
    )


def load_liar(path: Path) -> LoadedDataset:
    examples: list[Example] = []
    skipped = 0
    with path.open(encoding="utf-8", newline="") as handle:
        for cells in csv.reader(handle, delimiter="\t", quotechar='"'):
            parsed = _liar_row(cells)
            if parsed is None:
                if any(cell.strip() for cell in cells):
                    skipped += 1
                continue
            examples.append(parsed)
    if not examples:
        raise ValueError(f"{path} did not contain any LIAR rows with a known label.")
    return LoadedDataset(
        name="liar",
        domain=DOMAIN_NEWS,
        version="LIAR v1.0 test",
        source_url=LIAR_URL,
        sha256=sha256_file(path),
        split="test",
        license_note=(
            "LIAR v1.0 (Wang, ACL 2017), test.tsv. Research use only. "
            "PolitiFact retains copyright in the statements. This is not an open-source "
            "license. The file is cached and not committed."
        ),
        mapping_notes=LIAR_MAPPING_NOTES,
        examples=tuple(examples),
        skipped=skipped,
    )


def load_citations(path: Path | None = None) -> LoadedDataset:
    file = path or citations_path()
    examples: list[Example] = []
    skipped = 0
    with file.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parsed = _citation_row(line)
            if parsed is None:
                skipped += 1
                continue
            examples.append(parsed)
    if not examples:
        raise ValueError(f"{file} did not contain any citation rows.")
    return LoadedDataset(
        name="citations",
        domain=DOMAIN_CITATIONS,
        version="handwritten-1",
        source_url=str(file),
        sha256=sha256_file(file),
        split="handwritten",
        license_note=(
            "Hand-written for this repository. Real rows name works or pages that exist. "
            "Refuted rows were invented for the benchmark. Unverified rows are real works "
            "cited with a wrong detail on purpose."
        ),
        mapping_notes=CITATION_MAPPING_NOTES,
        examples=tuple(examples),
        skipped=skipped,
    )


def ensure_fever(cache: Path) -> LoadedDataset:
    path = cache / "fever" / "shared_task_dev.jsonl"
    download(FEVER_URL, path)
    _require_jsonl(path, "FEVER")
    return load_fever(path)


def ensure_scifact(cache: Path) -> LoadedDataset:
    root = cache / "scifact"
    archive = root / "data.tar.gz"
    claims = root / "claims_dev.jsonl"
    corpus = root / "corpus.jsonl"
    download(SCIFACT_URL, archive)
    if not claims.exists():
        _extract_member(archive, "claims_dev.jsonl", claims)
    corpus_sha = None
    if not corpus.exists():
        _extract_member(archive, "corpus.jsonl", corpus)
    if corpus.exists():
        corpus_sha = sha256_file(corpus)
    return load_scifact_claims(claims, corpus_sha256=corpus_sha)


def ensure_liar(cache: Path) -> LoadedDataset:
    root = cache / "liar"
    archive = root / "liar_dataset.zip"
    table = root / "test.tsv"
    download(LIAR_URL, archive)
    if not table.exists():
        _extract_liar_test(archive, table)
    return load_liar(table)


def _jsonl(
    path: Path,
    *,
    name: str,
    domain: str,
    version: str,
    source_url: str,
    split: str,
    license_note: str,
    mapping_notes: tuple[str, ...],
    parse_row: Any,
) -> LoadedDataset:
    examples: list[Example] = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            parsed = parse_row(line)
            if parsed is None:
                skipped += 1
                continue
            examples.append(parsed)
    if not examples:
        raise ValueError(f"{path} did not contain any {name} rows with a known label.")
    return LoadedDataset(
        name=name,
        domain=domain,
        version=version,
        source_url=source_url,
        sha256=sha256_file(path),
        split=split,
        license_note=license_note,
        mapping_notes=mapping_notes,
        examples=tuple(examples),
        skipped=skipped,
    )


def _fever_row(line: str) -> Example | None:
    raw = _object(line)
    if raw is None:
        return None
    label = raw.get("label")
    claim = raw.get("claim")
    ident = raw.get("id")
    if not isinstance(label, str) or label not in FEVER_LABELS or not isinstance(claim, str):
        return None
    if not claim.strip():
        return None
    row_id = f"fever-{ident}" if isinstance(ident, (str, int)) else f"fever-{_stable_id(claim)}"
    return Example(
        id=row_id,
        dataset="fever",
        domain=DOMAIN_GENERAL,
        text=claim.strip(),
        gold=BenchLabel(FEVER_LABELS[label]),
        raw_label=label,
    )


def _scifact_row(line: str) -> Example | None:
    raw = _object(line)
    if raw is None:
        return None
    claim = raw.get("claim")
    ident = raw.get("id")
    if not isinstance(claim, str) or not claim.strip():
        return None
    gold = scifact_gold(raw.get("evidence"))
    raw_label = _scifact_raw(raw.get("evidence"))
    row_id = f"scifact-{ident}" if isinstance(ident, (str, int)) else f"scifact-{_stable_id(claim)}"
    return Example(
        id=row_id,
        dataset="scifact",
        domain=DOMAIN_HEALTH,
        text=claim.strip(),
        gold=gold,
        raw_label=raw_label,
    )


def _scifact_raw(evidence: object) -> str:
    if not isinstance(evidence, dict) or not evidence:
        return "NO_EVIDENCE"
    labels: list[str] = []
    for rationales in evidence.values():
        if not isinstance(rationales, list):
            continue
        labels.extend(
            str(item["label"])
            for item in rationales
            if isinstance(item, dict) and isinstance(item.get("label"), str)
        )
    unique = sorted(set(labels))
    if not unique:
        return "NO_EVIDENCE"
    if len(unique) == 1:
        return unique[0]
    return "+".join(unique)


def _liar_row(cells: list[str]) -> Example | None:
    if len(cells) < 3:
        return None
    ident, label, statement = cells[0].strip(), cells[1].strip().casefold(), cells[2].strip()
    if label not in LIAR_LABELS or not statement:
        return None
    row_id = f"liar-{ident}" if ident else f"liar-{_stable_id(statement)}"
    return Example(
        id=row_id,
        dataset="liar",
        domain=DOMAIN_NEWS,
        text=statement,
        gold=BenchLabel(LIAR_LABELS[label]),
        raw_label=label,
    )


def _citation_row(line: str) -> Example | None:
    raw = _object(line)
    if raw is None:
        return None
    ident = raw.get("id")
    text = raw.get("text")
    label = raw.get("label")
    if not isinstance(ident, str) or not ident.strip():
        return None
    if not isinstance(text, str) or not text.strip() or not isinstance(label, str):
        return None
    return Example(
        id=ident.strip(),
        dataset="citations",
        domain=DOMAIN_CITATIONS,
        text=text.strip(),
        gold=bench_label(label),
        raw_label=label.strip(),
    )


def _object(line: str) -> dict[str, Any] | None:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None
    if isinstance(raw, dict):
        return raw
    return None


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _require_jsonl(path: Path, name: str) -> None:
    with path.open("rb") as handle:
        first = handle.read(1)
    if first != b"{":
        raise ValueError(f"{name} download at {path} is not JSONL. Delete it and retry.")


def _extract_member(archive: Path, name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        dest.write_bytes(_tar_member(tar, name))


def _tar_member(archive: tarfile.TarFile, name: str) -> bytes:
    for member in archive.getmembers():
        if not member.isfile() or ".." in Path(member.name).parts:
            continue
        if Path(member.name).name != name:
            continue
        extracted = archive.extractfile(member)
        if extracted is None:
            continue
        return extracted.read()
    raise ValueError(f"SciFact archive has no {name}.")


def _extract_liar_test(archive: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        chosen = ""
        for candidate in bundle.namelist():
            if ".." in Path(candidate).parts:
                continue
            if Path(candidate).name == "test.tsv":
                chosen = candidate
                break
        if not chosen:
            raise ValueError("LIAR zip has no test.tsv.")
        dest.write_bytes(bundle.read(chosen))


def read_tar_member(blob: bytes, name: str) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        return _tar_member(archive, name)
