# Contributing

VerifAI is a student research project: a self-hosted claim checker. Verdicts come from records retrieved at check time. It does not call a commercial model API (no OpenAI, Anthropic, Gemini, or similar). A local model is optional and is not a source.

## Evidence principle

No single source decides a verdict. A check records which providers were consulted, which ones contributed (including an authoritative not-found), and which ones were unavailable. A claim verdict other than unverified needs at least two independent sources that agree. Google Fact Check reviews count as one source. The same paper from two indexes counts as one source. The function is `corroborate` in `src/verifai/corroboration.py`.

The citation expert is constructed with doi.org, Crossref, DataCite, OpenAlex, a direct URL check, and the Wayback Machine. One index can still verify a citation when another is down, at lower confidence. That question is whether the reference exists, which is separate from whether the work supports a claim.

The statement lives in `EVIDENCE_PRINCIPLE` in `src/verifai/schemas.py` and is repeated on the `Expert` and `EvidenceProvider` interfaces.

## macOS setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (`brew install uv` works). Then, from this directory:

```bash
make install
make run
```

`make install` syncs the lockfile. `make run` starts the app on `127.0.0.1:8741`.

## Checks

```bash
make lint
make typecheck
make test
```

`make lint` is `ruff check`, `ruff format --check`, and `node --check` on the extension. `make typecheck` is `mypy` in strict mode. `make test` runs the extension tests and fails if branch coverage of `src/verifai` drops below 85%. `pre-commit install` runs the ruff and mypy hooks.

Tests mock HTTP. They do not call the public APIs. `mypy` runs in strict mode. Ruff uses the rule set in `pyproject.toml`.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `chore:`, `test:`, `ci:`, `docs:`, `refactor:`. The subject says what changed. The body says why.

Do not commit secrets, `.env`, or API keys. Optional keys stay in the environment.

## Pull requests

Describe the behavior change and how you tested it. A change to matching or to a provider should keep the evidence principle intact: the new source is not a sole decider, and an outage is not treated as a fabricated citation.
