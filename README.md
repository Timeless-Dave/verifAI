# VerifAI

VerifAI is a self-hosted tool that checks whether the citations in a pasted answer actually exist. It is a student research project. This version is the citation checker: the first expert in a larger design where a pasted claim, chatbot answer, or link is routed to domain checkers.

A verdict is computed from records retrieved at check time. It is never taken from a language model’s memory, and this version does not call a commercial model API (no OpenAI, Anthropic, Gemini, or similar). Free public data APIs are used: doi.org, Crossref, OpenAlex, DataCite, and the Wayback Machine. Semantic Scholar is optional and stays off unless you set an API key.

The check answers a narrow question: does this DOI, URL, or bibliographic reference correspond to a real work or page, and do the authors, year, and venue printed next to it agree with that record? It does not read the paper and it does not decide whether the source supports the paragraph around it.

## Evidence principle

Verdicts aggregate evidence from multiple independent sources, record which sources were consulted and which ones contributed, and still come back when any single provider is unavailable. No provider is the sole decider.

The Google Fact Check Tools API, if it is added later, is one more provider under this rule. It must never be the only source a verdict depends on. The same rule is part of the `Expert` interface in `src/verifai/experts/base.py` and of the `EvidenceProvider` protocol in `src/verifai/evidence/base.py`.

For citations, that means:

- Free-text references are searched in **both** Crossref and OpenAlex. A strong match from one index can still verify a citation when the other is down or has no comparable record, at lower confidence, and the result names the gap.
- **DataCite** is searched as well, with the cited year and first author. Some real works, including arXiv preprints, are registered there and do not show up in Crossref or OpenAlex title search, which may only return a later reprint of the same title. A year-matching DataCite record is preferred over that reprint. DataCite is not a sole decider: if it is down, the check continues on Crossref and OpenAlex.
- DOIs are resolved at **doi.org** (the registry resolver, which delegates to Crossref, DataCite, or another agency) and cross-checked in **OpenAlex**. doi.org and the Crossref REST API are not two independent copies of the same registry. If doi.org is down, the Crossref REST API is the fallback for that lookup, and OpenAlex remains the independent cross-check.
- URLs get a direct fetch. The Wayback Machine is the fallback when that fetch does not confirm the page.
- Semantic Scholar is a third opinion only when `VERIFAI_SEMANTIC_SCHOLAR_API_KEY` is set. It is never called without a key, because the public endpoint rate-limits hard, and it is never the only provider.

“Contributed” means the provider returned a definite answer, including an authoritative not-found. “Unavailable” means a timeout, HTTP 5xx, or network error. An outage is not treated as evidence that a citation is fake.

## What you need

A Mac (including Apple silicon) or any machine with a recent Python. This version does not load a model. The citation checker is HTTP plus deterministic string matching, so 24 GB of RAM is far more than it needs. `uv` installs a CPython if the system one is older than 3.11. RapidFuzz ships wheels, so you do not need a compiler.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On macOS, `brew install uv` works too.

### macOS

From this directory:

```bash
cp .env.example .env
# Put your real email in VERIFAI_MAILTO so Crossref's polite pool will accept the client.
make install
make run
```

`make install` runs `uv sync --all-groups` and creates the local environment.
`make run` installs if that has not been done and starts the app on `127.0.0.1:8741`.

Then open [http://127.0.0.1:8741](http://127.0.0.1:8741). After uv itself is installed, those two commands are the setup.

`make test` runs the unit tests. They mock HTTP and do not call the public APIs.

Useful equivalents:

```bash
uv sync --all-groups
uv run verifai
uv run pytest
```

The server reads `VERIFAI_HOST` and `VERIFAI_PORT` (defaults `127.0.0.1` and `8741`).

### Try the example

The page has **Load an example**. It pastes a short chatbot-style answer with:

- Vaswani et al., 2017, *Attention Is All You Need* (real)
- Jinek et al., 2012, *Science*, the CRISPR endonuclease paper (real)
- a real PLOS ONE DOI
- an invented sourdough paper
- an unregistered DOI
- a Wikipedia URL that 404s

You should see verified rows for the real works and not-found rows for the invented ones. The first call is slower because it waits on the public APIs. Repeat checks are cached in memory for an hour (ten minutes for 404s).

### JSON API

Interactive docs: [http://127.0.0.1:8741/docs](http://127.0.0.1:8741/docs).

```bash
curl -s http://127.0.0.1:8741/api/v1/check \
  -H 'content-type: application/json' \
  -d '{"text":"https://doi.org/10.1126/science.1225829"}'
```

`GET /health` lists the providers that were actually configured.

Each citation is one of:

| Status | Meaning |
| --- | --- |
| `verified` | A retrieved record matches, the URL loaded (HTTP 2xx/3xx), or Wayback has a snapshot. |
| `partially_matches` | A real record is close, but authors, year, or venue disagree, or a live DOI points at a different title than the one printed beside it. |
| `not_found` | The sources that answered have nothing close enough, or the URL is a clear 404 with no archive. Often a fabricated citation. Less certain when a source was down. |
| `unreachable` | Timeout, server error, or the host refused the fetch. Not evidence the citation is fake. |

The top-level `verdict` (`supported`, `mixed`, `unsupported`, `unverifiable`) summarizes those rows. `verdict_scope` is `citations`. `sources_consulted`, `sources_contributed`, and `sources_unavailable` say who was asked.

Confidence is confidence in that status, from 0 to 1, not a probability that a claim is true.

## Layout

```
src/verifai/
  main.py                 FastAPI app
  config.py               VERIFAI_* settings
  schemas.py              ExpertResult, verdicts, evidence
  http.py                 User-Agent, timeouts, retries, cache
  evidence/               pluggable provider protocol and registry
  experts/base.py         Expert interface
  experts/router.py       stub router (citation expert only, for now)
  experts/citation/       extractor, matcher, decider, providers
  web/                    Jinja templates, HTMX, static CSS
tests/                    pytest, respx-mocked HTTP, fixture responses
```

The router is a stub. Later, health, news, and general experts can be registered beside the citation expert. One claim may go to several experts. Their results stay separate so no single expert is the sole decider.

Extraction is regular expressions only. It looks for DOIs (including `https://doi.org/...`), http(s) URLs, APA-like `Author, A. (Year). Title. Venue.` lines, and `Surname et al., Year, Venue, Title`. A bare `(Smith et al., 2021)` is skipped. Without a title it is not specific enough to search without calling a real paper fabricated.

Matching uses RapidFuzz on normalized titles (case, accents, and punctuation folded). Author surnames, year, and venue are compared field by field. When both indexes return the same work, confidence goes up. When they return different works at similar strength, the status stays at partially matches. An off-by-one year is reported as a partial match because preprint and publication years often differ; the explanation says so. If one index has the cited year and the other has a later edition of the same title, the edition that agrees with the citation wins, and the other year is mentioned rather than failing the row.

## Configuration

| Variable | Default | Role |
| --- | --- | --- |
| `VERIFAI_MAILTO` | `verifai-dev@example.com` | Contact address in the User-Agent. Crossref’s polite pool wants a real mailbox. Also sent as OpenAlex `mailto`. |
| `VERIFAI_OPENALEX_API_KEY` | empty | Optional. Without a key, OpenAlex shares a small free daily budget per IP. A 402 or 429 becomes “OpenAlex unavailable”, and Crossref still answers. |
| `VERIFAI_SEMANTIC_SCHOLAR_API_KEY` | empty | Optional. Unset means Semantic Scholar is not called. |
| `VERIFAI_HTTP_TIMEOUT` | `20` | Seconds to wait for a response body. Connecting still gives up after 5 seconds. |
| `VERIFAI_HTTP_RETRIES` | `1` | Extra attempts on timeout, 429, and 5xx. |
| `VERIFAI_CACHE_TTL_SECONDS` | `3600` | In-memory cache for successful responses. |
| `VERIFAI_HOST`, `VERIFAI_PORT` | `127.0.0.1`, `8741` | Bind address. |

The process User-Agent looks like `VerifAI/0.1.0 (mailto:you@example.com)`.

## Tests and CI

```bash
make test
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same pytest suite on push and pull request. Tests do not touch the network. Provider payloads in `tests/fakes.py` follow the JSON shapes returned by doi.org, Crossref, OpenAlex, and DataCite.

## Known limitations

- Existence is not endorsement. A verified citation can still be irrelevant to the claim, misquoted, or retracted. Retraction checks are not in this version.
- The extractor misses unusual styles, references wrapped in ways that do not survive joining lines, and titles that contain abbreviations with periods (`U.S.`). It ignores author-year mentions that have no title.
- Crossref’s search returns loosely related works. Those are kept only when the title is actually close. A fabricated title that is a light edit of a real one can come back as partially matches rather than not found.
- Crossref and OpenAlex title search can rank a later reprint above the original. “Attention Is All You Need” (Vaswani et al., 2017) is the case this checker was measured against: those indexes’ top hits are 2025 copies of the same title, while DataCite still has the 2017 arXiv record (`10.48550/arxiv.1706.03762`). When DataCite is unavailable, that citation comes back as a year mismatch against the reprint rather than as verified.
- Off-by-one years are partial matches even when the rest agrees. That is deliberate and sometimes noisy.
- Venue abbreviations are recognized when they are initials (`PNAS`). Nicknames that are not initials (`NeurIPS` versus the full conference name) can be flagged as a venue mismatch.
- OpenAlex without an API key draws on a shared per-IP daily budget. When that budget is spent, OpenAlex is reported unavailable and the check continues on Crossref and doi.org at lower confidence.
- doi.org and Crossref are the same registry for Crossref DOIs. Independence comes from OpenAlex (and Semantic Scholar, if configured), not from calling both doi.org and Crossref for a DOI that doi.org already resolved.
- URL checks use an identifiable User-Agent, not a browser disguise. Sites that block unknown clients come back unreachable (HTTP 401/403), which is not a fabrication finding. Soft 404 pages that return HTTP 200 look verified.
- Private, loopback, and link-local URLs, and URLs with an embedded password, are not fetched.
- The cache is in memory only. Restarting the process clears it. At most 25 citations are checked per paste, to keep outbound traffic polite.
- There is no account system, database, or browser extension. Those are later work. No local model is wired up; the checker is required to work with none.
- This is not a substitute for opening the paper.

## License

MIT. See `LICENSE`.

The page vendors [htmx 2.0.4](https://htmx.org/) (`src/verifai/web/static/htmx.min.js`) under the Zero-Clause BSD license in `src/verifai/web/static/htmx.LICENSE`.
