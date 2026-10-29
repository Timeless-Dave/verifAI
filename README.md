# VerifAI

VerifAI is a self-hosted tool that checks claims for everyday people. Paste a sentence, a chatbot answer, or a link. A router sends each claim to the checker that fits it, and the page shows one verdict, a confidence, a plain-language explanation, and every source as a link.

A verdict is computed from records retrieved at check time. It is never taken from a language model’s memory, and VerifAI does not call a commercial model API (no OpenAI, Anthropic, Gemini, or similar). Optional local models (MLX, Ollama, llama.cpp, or a small NLI model) only judge passages that were just retrieved. Free public data APIs supply the evidence.

The citation checker is still here. It answers a narrower question: does this DOI, URL, or bibliographic reference correspond to a real work or page, and do the authors, year, and venue printed next to it agree with that record? It does not read the paper. The other experts ask whether retrieved evidence supports or contradicts the claim itself.

## Evidence principle

Verdicts aggregate evidence from multiple independent sources, record which sources were consulted and which ones contributed, and still come back when any single provider is unavailable. No provider is the sole decider.

For a claim, a verdict other than unverified (insufficient evidence) needs at least two independent sources that agree. Google Fact Check reviews, however many come back, count as one source. The same paper found in two indexes counts as one source. One source on each side is not agreement. The rule lives in `src/verifai/corroboration.py` and is applied by every claim expert. It is also stated on the `Expert` interface in `src/verifai/experts/base.py` and on the `EvidenceProvider` protocol in `src/verifai/evidence/base.py`.

For citations, that means:

- Free-text references are searched in **both** Crossref and OpenAlex. A strong match from one index can still verify a citation when the other is down or has no comparable record, at lower confidence, and the result names the gap.
- **DataCite** is searched as well, with the cited year and first author. Some real works, including arXiv preprints, are registered there and do not show up in Crossref or OpenAlex title search, which may only return a later reprint of the same title. A year-matching DataCite record is preferred over that reprint. DataCite is not a sole decider: if it is down, the check continues on Crossref and OpenAlex.
- DOIs are resolved at **doi.org** (the registry resolver, which delegates to Crossref, DataCite, or another agency) and cross-checked in **OpenAlex**. doi.org and the Crossref REST API are not two independent copies of the same registry. If doi.org is down, the Crossref REST API is the fallback for that lookup, and OpenAlex remains the independent cross-check.
- URLs get a direct fetch. The Wayback Machine is the fallback when that fetch does not confirm the page.
- Semantic Scholar is a third opinion only when `VERIFAI_SEMANTIC_SCHOLAR_API_KEY` is set. It is never called without a key, because the public endpoint rate-limits hard, and it is never the only provider.

“Contributed” means the provider returned a definite answer, including an authoritative not-found. “Unavailable” means a timeout, HTTP 5xx, or network error. An outage is not treated as evidence that a citation is fake.

## What you need

A Mac (including a 24 GB M5 Pro) or any machine with a recent Python. The citation checker is HTTP plus deterministic string matching and does not need a model. Claim judgments use a local model when one is installed, and they still return an unverified result when it is not. `uv` installs a CPython if the system one is older than 3.11. RapidFuzz ships wheels, so you do not need a compiler.

On a 24 GB Apple silicon Mac the intended default is MLX with a 4-bit Qwen3.5 9B (`mlx-community/Qwen3.5-9B-4bit`, about 6 GB of weights). That leaves room for the app, a small DeBERTa NLI model, and ordinary desktop use. A 7B 4-bit model is the fallback if you want more headroom. Nothing is downloaded until you install a backend and run a check.

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

`make test` runs the unit tests and fails if coverage of `src/verifai` drops below 85%. The tests mock HTTP and do not call the public APIs.

Useful equivalents:

```bash
uv sync --all-groups
uv run verifai
uv run pytest
```

The server reads `VERIFAI_HOST` and `VERIFAI_PORT` (defaults `127.0.0.1` and `8741`).

### Local model on a 24 GB Mac

`VERIFAI_MODEL_BACKEND` defaults to `auto`. It uses the first backend that is actually installed:

1. A small NLI model (DeBERTa MNLI, or a MiniCheck-style classifier if you set that model id) when `transformers` imports.
2. MLX on Apple silicon when `mlx_lm` imports. The default weights are `mlx-community/Qwen3.5-9B-4bit`.
3. Ollama at `http://127.0.0.1:11434`.
4. A llama.cpp server at `http://127.0.0.1:8080/v1` (OpenAI-compatible chat completions).

Set `VERIFAI_MODEL_BACKEND=none` to skip judging. Fact-check ratings are still read. Pages and abstracts are listed and linked, and the verdict stays unverified.

MLX, on Apple silicon:

```bash
uv pip install mlx-lm
# The first judgment downloads the model into the Hugging Face cache.
```

If 9B is tight next to other apps:

```bash
# VERIFAI_MLX_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit
```

NLI (entailment, lighter than a 9B model, and preferred by `auto` when it is installed):

```bash
uv pip install torch transformers
```

The default id is `MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli`. For a MiniCheck-style model set `VERIFAI_NLI_MODEL` to that model’s id (for example `lytang/MiniCheck-DeBERTa-v3-Large`). The larger checkpoint needs more memory than the small DeBERTa.

Ollama:

```bash
brew install ollama
ollama pull qwen3.5:9b
```

Then set `VERIFAI_MODEL_BACKEND=ollama`.

llama.cpp: start `llama-server` with a GGUF (a Q4 quant of a 9B model fits the same 24 GB machine) and set `VERIFAI_MODEL_BACKEND=llamacpp`. The base URL is `VERIFAI_LLAMACPP_BASE_URL` (default `http://127.0.0.1:8080/v1`).

These packages are optional. CI and `make test` do not install or download them. Tests use fakes.

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

`GET /health` lists the providers that were actually configured, the experts, and whether a local model is loaded.

Claim experts:

| Expert | Evidence | When it runs |
| --- | --- | --- |
| General | Wikipedia, Wikidata, pages linked in the paste, and Google Fact Check reviews if `VERIFAI_GOOGLE_FACTCHECK_API_KEY` is set | A factual sentence that is not specifically health or news |
| Health and science | PubMed (NCBI E-utilities) and Europe PMC. Systematic reviews and meta-analyses are requested first | Medical or scientific wording. Every result says this is not medical advice |
| News and politics | Google Fact Check reviews, GDELT articles (each outlet is its own source), Wikipedia current events, and fetched article text | Elections, governments, and similar wording |
| Citation | doi.org, Crossref, OpenAlex, DataCite, a direct fetch, and the Wayback Machine | A DOI, URL, year, or “et al.” |

A paste can go to more than one expert. The page leads with the combined verdict. Health text is not a diagnosis.

Each citation is one of:

| Status | Meaning |
| --- | --- |
| `verified` | A retrieved record matches, the URL loaded (HTTP 2xx/3xx), or Wayback has a snapshot. |
| `partially_matches` | A real record is close, but authors, year, or venue disagree, or a live DOI points at a different title than the one printed beside it. |
| `not_found` | The sources that answered have nothing close enough, or the URL is a clear 404 with no archive. Often a fabricated citation. Less certain when a source was down. |
| `unreachable` | Timeout, server error, or the host refused the fetch. Not evidence the citation is fake. |

The top-level `verdict` (`supported`, `mixed`, `unsupported`, `unverifiable`) summarizes those rows. `verdict_scope` is `citations`. `sources_consulted`, `sources_contributed`, and `sources_unavailable` say who was asked.

Confidence is confidence in that status, from 0 to 1, not a probability that a claim is true.

## Browser extension

`extension/` is a Manifest V3 extension for Chrome and Firefox. It stays quiet on ordinary pages. A small badge appears when the page looks like news, a health article, a chatbot transcript, or a page with citations (a DOI, or repeated “et al.”). The decision is a local heuristic in `extension/src/heuristic.js`. Nothing is sent until you click the badge.

The badge checks the selection, or a few claim-like sentences, against the local server. The default URL is `http://127.0.0.1:8741`. Change it from the extension’s options page. The manifest only permits requests to `127.0.0.1` and `localhost`; point the server at another host only after you add that origin to `host_permissions`.

There is no build step. From this directory:

```bash
make extension-test
```

Load it unpacked:

1. Chrome: open `chrome://extensions`, turn on Developer mode, choose “Load unpacked”, and select the `extension/` folder.
2. Firefox: open `about:debugging#/runtime/this-firefox`, choose “Load Temporary Add-on…”, and select `extension/manifest.json`.

Start VerifAI first (`make run`), then open a news article or a chatbot transcript. A coding site or a short recipe page should show no badge.

## Layout

```
src/verifai/
  main.py                 FastAPI app
  config.py               VERIFAI_* settings
  schemas.py              ExpertResult, verdicts, evidence
  corroboration.py        two-source rule, including the Google cap
  claims.py               rule-based claim splitter
  orchestrate.py          per-claim routing and the combined verdict
  http.py                 User-Agent, timeouts, retries, cache
  models/                 MLX, Ollama, llama.cpp, and NLI judges
  retrieval/              Wikipedia, Wikidata, PubMed, Europe PMC, GDELT, pages
  evidence/               pluggable provider protocol and registry
  experts/                citation, general, health, and news checkers
  web/                    Jinja templates, HTMX, static CSS
extension/                Manifest V3 extension and its heuristic tests
tests/                    pytest, respx-mocked HTTP, fixture responses
```

One claim may go to several experts. Their evidence is combined by the same corroboration rule, so no single expert or provider is the sole decider.

Extraction is regular expressions only. It looks for DOIs (including `https://doi.org/...`), http(s) URLs, APA-like `Author, A. (Year). Title. Venue.` lines, and `Surname et al., Year, Venue, Title`. A bare `(Smith et al., 2021)` is skipped. Without a title it is not specific enough to search without calling a real paper fabricated.

Matching uses RapidFuzz on normalized titles (case, accents, and punctuation folded). Author surnames, year, and venue are compared field by field. When both indexes return the same work, confidence goes up. When they return different works at similar strength, the status stays at partially matches. An off-by-one year is reported as a partial match because preprint and publication years often differ; the explanation says so. If one index has the cited year and the other has a later edition of the same title, the edition that agrees with the citation wins, and the other year is mentioned rather than failing the row.

## Configuration

| Variable | Default | Role |
| --- | --- | --- |
| `VERIFAI_MAILTO` | `verifai-dev@example.com` | Contact address in the User-Agent. Crossref’s polite pool wants a real mailbox. Also sent as OpenAlex `mailto`. |
| `VERIFAI_OPENALEX_API_KEY` | empty | Optional. Without a key, OpenAlex shares a small free daily budget per IP. A 402 or 429 becomes “OpenAlex unavailable”, and Crossref still answers. |
| `VERIFAI_SEMANTIC_SCHOLAR_API_KEY` | empty | Optional. Unset means Semantic Scholar is not called. |
| `VERIFAI_HTTP_TIMEOUT` | `20` | Seconds to wait for a response body. Connecting still gives up after 5 seconds. Values above 120 are rejected. |
| `VERIFAI_HTTP_RETRIES` | `1` | Extra attempts on timeout, 429, and 5xx. Capped at 8. |
| `VERIFAI_HTTP_MIN_INTERVAL_SECONDS` | `0.2` | Minimum gap between requests to the same host. `0` disables it. Other hosts are not blocked. |
| `VERIFAI_HTTP_MAX_RESPONSE_BYTES` | `2000000` | Larger response bodies are discarded and that provider is unavailable. |
| `VERIFAI_MAX_TEXT_CHARS` | `50000` | Longest paste that will be checked. The HTTP body limit is a bit more than six times this, to allow JSON escaping. |
| `VERIFAI_CACHE_TTL_SECONDS` | `3600` | In-memory cache for successful responses. |
| `VERIFAI_HOST`, `VERIFAI_PORT` | `127.0.0.1`, `8741` | Bind address. |
| `VERIFAI_GOOGLE_FACTCHECK_API_KEY` | empty | Optional. Unset means Google Fact Check is not called. Reviews still count as one source. |
| `VERIFAI_MODEL_BACKEND` | `auto` | `auto`, `mlx`, `ollama`, `llamacpp`, `nli`, or `none`. |
| `VERIFAI_MLX_MODEL` | `mlx-community/Qwen3.5-9B-4bit` | MLX weights. Used on Apple silicon when that backend is selected. |
| `VERIFAI_OLLAMA_BASE_URL`, `VERIFAI_OLLAMA_MODEL` | `http://127.0.0.1:11434`, `qwen3.5:9b` | Ollama. |
| `VERIFAI_LLAMACPP_BASE_URL`, `VERIFAI_LLAMACPP_MODEL` | `http://127.0.0.1:8080/v1`, `qwen3.5-9b` | llama.cpp server. |
| `VERIFAI_NLI_MODEL` | `MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli` | Entailment model. MiniCheck-style ids are accepted. |
| `VERIFAI_NLI_ENABLED` | `true` | When false, `auto` skips NLI even if `transformers` is installed. |
| `VERIFAI_MAX_CLAIMS` | `5` | How many extracted claims are checked per paste. |

The process User-Agent looks like `VerifAI/0.2.0 (mailto:you@example.com)`.

## Tests and CI

```bash
make lint
make typecheck
make test
```

`make lint` runs ruff and `node --check` on the extension. `make typecheck` runs mypy in strict mode. `make test` runs the extension tests and pytest, and fails if branch coverage of `src/verifai` drops below 85%.

GitHub Actions (`.github/workflows/ci.yml`) runs the Python checks on 3.11, 3.12, and 3.13, and runs the extension lint and tests once. Tests do not touch the network or download a model. Provider payloads in `tests/fakes.py` follow the JSON shapes returned by doi.org, Crossref, OpenAlex, and DataCite. Claim providers are stubbed the same way.

`pre-commit install` runs the same ruff and mypy checks on each commit. The hooks call `uv run`, so `make install` has to have succeeded first.

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
- There is no account system or database. The browser extension only talks to the server you configure, defaulting to this machine.
- Health results are not medical advice. They report what the retrieved studies say.
- A local model can misread a passage. The corroboration rule still requires two independent sources, and a model is not itself a source.
- Wikipedia and Wikidata are different databases and count separately. Google Fact Check does not: every review it returns is one source.
- PubMed and Europe PMC overlap. The same DOI or PMID is one paper, not two sources. Two different reviews can agree.
- GDELT is an index. Each news outlet counts as its own source, and the page names who said what.
- This is not a substitute for opening the paper or the article.

## Contributing and security

See `CONTRIBUTING.md` for the evidence principle and local checks, and `SECURITY.md` for how to report a vulnerability. Do not put secrets in issues or pull requests.

## License

MIT. See `LICENSE`.

The page vendors [htmx 2.0.4](https://htmx.org/) (`src/verifai/web/static/htmx.min.js`) under the Zero-Clause BSD license in `src/verifai/web/static/htmx.LICENSE`.
