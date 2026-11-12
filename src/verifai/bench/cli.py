"""``python -m verifai.bench``. ``make bench`` calls this. CI does not.

The default is backend ``none``, 50 rows, seed 0. That downloads datasets
on first use and does not load a model. Claim rows still go through
``run_check``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

import httpx

from verifai.bench.datasets import (
    LoadedDataset,
    default_cache_dir,
    ensure_fever,
    ensure_liar,
    ensure_scifact,
    load_citations,
    load_fever,
    load_liar,
    load_scifact_claims,
    sha256_file,
)
from verifai.bench.fixtures import FixtureRetriever, load_passage_fixtures
from verifai.bench.report import RunMeta, git_sha, utc_now, write_report
from verifai.bench.scoring import evaluate
from verifai.config import Settings
from verifai.experts.citation import build_citation_expert
from verifai.experts.general import GeneralExpert, build_general_expert
from verifai.experts.health import HealthExpert, build_health_expert
from verifai.experts.news import NewsExpert, build_news_expert
from verifai.experts.router import Router
from verifai.http import PoliteHttp
from verifai.models.judge import Judge, build_judge
from verifai.retrieval.types import Passage

_DATASETS = ("fever", "scifact", "liar", "citations")
_CLAIM_DATASETS = ("fever", "scifact", "liar")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError, httpx.HTTPError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verifai.bench",
        description=(
            "Score VerifAI on labeled claims by calling run_check. "
            "Datasets download into the cache. Results go to results/ and are not committed."
        ),
    )
    parser.add_argument("--datasets", default=",".join(_DATASETS))
    parser.add_argument(
        "--backend",
        default="none",
        choices=("auto", "mlx", "ollama", "llamacpp", "nli", "none"),
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--fever", type=Path, default=None)
    parser.add_argument("--scifact", type=Path, default=None)
    parser.add_argument("--liar", type=Path, default=None)
    parser.add_argument("--citations", type=Path, default=None)
    parser.add_argument(
        "--passage-fixtures",
        type=Path,
        default=None,
        help="JSON object of claim text to passages. Required for offline claim rows.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    selected = _parse_datasets(args.datasets)
    if args.limit < 0:
        raise ValueError("--limit must be 0 (all rows) or a positive sample size.")
    if args.offline and "citations" in selected:
        raise ValueError(
            "Offline mode does not score citations. The citation checker calls live "
            "bibliographic APIs. Drop citations, or run without --offline."
        )
    claims = [name for name in selected if name in _CLAIM_DATASETS]
    if args.offline and claims and args.passage_fixtures is None:
        raise ValueError("Offline claim runs need --passage-fixtures.")
    settings = _settings(args.backend, args.model_id)
    cache = args.cache_dir or default_cache_dir()
    limit = None if args.limit == 0 else args.limit
    # Record the commit before the run. Code can change during a long run.
    started = utc_now()
    sha = git_sha(_repo_root())
    datasets = _load_datasets(args, selected, cache)
    judge = build_judge(settings)
    http: PoliteHttp | None = None
    try:
        if args.offline:
            if args.passage_fixtures is None:
                raise ValueError("Offline claim runs need --passage-fixtures.")
            router = _fixture_router(judge, load_passage_fixtures(args.passage_fixtures))
        else:
            http = PoliteHttp(settings)
            router = _live_router(http, settings, judge)
        result = evaluate(router, datasets, seed=args.seed, limit=limit)
    finally:
        if http is not None:
            http.close()
    json_path, markdown_path = _output_paths(args, settings.model_backend, args.seed, limit)
    meta = RunMeta(
        date=started,
        git_sha=sha,
        backend=settings.model_backend,
        model_id=_model_id(settings),
        judge_name=getattr(judge, "name", settings.model_backend),
        seed=args.seed,
        limit=limit,
        offline=bool(args.offline),
        google_factcheck=bool(settings.google_factcheck_api_key) and not args.offline,
    )
    write_report(result, meta, json_path=json_path, markdown_path=markdown_path)
    sys.stdout.write(
        f"Wrote {json_path}\nWrote {markdown_path}\n"
        f"Examples {result.overall.n}; accuracy {_fmt(result.overall.accuracy)}; "
        f"macro-F1 {_fmt(result.overall.macro_f1)}; "
        f"abstention {_fmt(result.overall.abstention_rate)}\n"
    )
    return 0


def _settings(backend: str, model_id: str | None) -> Settings:
    if model_id and backend == "none":
        raise ValueError("The none backend does not take a model id.")
    if model_id and backend == "auto":
        raise ValueError("--model-id needs a specific backend, not auto.")
    chosen = cast("Literal['auto', 'mlx', 'ollama', 'llamacpp', 'nli', 'none']", backend)
    if chosen == "nli" and model_id:
        return Settings(model_backend="nli", nli_model=model_id, max_concurrency=2)
    if chosen == "mlx" and model_id:
        return Settings(model_backend="mlx", mlx_model=model_id, max_concurrency=2)
    if chosen == "ollama" and model_id:
        return Settings(model_backend="ollama", ollama_model=model_id, max_concurrency=2)
    if chosen == "llamacpp" and model_id:
        return Settings(model_backend="llamacpp", llamacpp_model=model_id, max_concurrency=2)
    return Settings(model_backend=chosen, max_concurrency=2)


def _model_id(settings: Settings) -> str:
    if settings.model_backend == "none":
        return "none"
    if settings.model_backend == "nli":
        return settings.nli_model
    if settings.model_backend == "mlx":
        return settings.mlx_model
    if settings.model_backend == "ollama":
        return settings.ollama_model
    if settings.model_backend == "llamacpp":
        return settings.llamacpp_model
    return "auto"


def _live_router(http: PoliteHttp, settings: Settings, judge: Judge) -> Router:
    return Router(
        [
            build_citation_expert(http, settings),
            build_general_expert(http, settings, judge),
            build_health_expert(http, settings, judge),
            build_news_expert(http, settings, judge),
        ]
    )


def _fixture_router(judge: Judge, passages: dict[str, tuple[Passage, ...]]) -> Router:
    fixture = FixtureRetriever(passages)
    return Router(
        [
            GeneralExpert([fixture], judge, google_configured=False),
            HealthExpert([fixture], judge),
            NewsExpert([fixture], judge, google_configured=False),
        ]
    )


def _parse_datasets(raw: str) -> list[str]:
    names = [part.strip().casefold() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("Choose at least one dataset.")
    unknown = [name for name in names if name not in _DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset {unknown[0]!r}. Choose from {', '.join(_DATASETS)}.")
    seen: list[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def _load_datasets(
    args: argparse.Namespace, selected: list[str], cache: Path
) -> list[LoadedDataset]:
    loaded: list[LoadedDataset] = []
    for name in selected:
        if name == "fever":
            loaded.append(_load_fever(args, cache))
        elif name == "scifact":
            loaded.append(_load_scifact(args, cache))
        elif name == "liar":
            loaded.append(_load_liar(args, cache))
        else:
            loaded.append(load_citations(args.citations))
    return loaded


def _load_fever(args: argparse.Namespace, cache: Path) -> LoadedDataset:
    if args.fever is not None:
        return load_fever(args.fever)
    if args.offline:
        path = cache / "fever" / "shared_task_dev.jsonl"
        if not path.exists():
            raise ValueError(f"Offline FEVER file is missing: {path}. Pass --fever.")
        return load_fever(path)
    return ensure_fever(cache)


def _load_scifact(args: argparse.Namespace, cache: Path) -> LoadedDataset:
    if args.scifact is not None:
        return load_scifact_claims(args.scifact)
    if args.offline:
        path = cache / "scifact" / "claims_dev.jsonl"
        if not path.exists():
            raise ValueError(f"Offline SciFact claims are missing: {path}. Pass --scifact.")
        corpus = cache / "scifact" / "corpus.jsonl"
        corpus_sha = sha256_file(corpus) if corpus.exists() else None
        return load_scifact_claims(path, corpus_sha256=corpus_sha)
    return ensure_scifact(cache)


def _load_liar(args: argparse.Namespace, cache: Path) -> LoadedDataset:
    if args.liar is not None:
        return load_liar(args.liar)
    if args.offline:
        path = cache / "liar" / "test.tsv"
        if not path.exists():
            raise ValueError(f"Offline LIAR file is missing: {path}. Pass --liar.")
        return load_liar(path)
    return ensure_liar(cache)


def _output_paths(
    args: argparse.Namespace,
    backend: str,
    seed: int,
    limit: int | None,
) -> tuple[Path, Path]:
    if args.json_out is not None and args.markdown_out is not None:
        return args.json_out, args.markdown_out
    stamp = utc_now().replace(":", "").replace("-", "")
    limit_bit = "all" if limit is None else str(limit)
    directory = Path("results")
    stem = f"bench-{stamp}-{backend}-seed{seed}-n{limit_bit}"
    return args.json_out or (directory / f"{stem}.json"), args.markdown_out or (
        directory / f"{stem}.md"
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"
