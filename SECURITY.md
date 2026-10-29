# Security policy

VerifAI is a student research project. It is self-hosted. It does not ship accounts, and it does not call a commercial model API.

## Reporting a vulnerability

Please use [GitHub private vulnerability reporting](https://github.com/Timeless-Dave/verifAI/security/advisories/new) for this repository. Do not open a public issue for a security report, and do not include exploit payloads, credentials, or API keys in the report.

## What this process will fetch

A check can send queries to doi.org, Crossref, OpenAlex, DataCite, Wikipedia, Wikidata, PubMed, Europe PMC, GDELT, and the Wayback Machine, and it may fetch a URL the user pasted. Semantic Scholar is contacted only when `VERIFAI_SEMANTIC_SCHOLAR_API_KEY` is set. Google Fact Check is contacted only when `VERIFAI_GOOGLE_FACTCHECK_API_KEY` is set. A local model, when configured, is called on this machine (MLX in-process, or Ollama / llama.cpp on localhost). Those remote requests leave the machine. Do not paste text you would not send to those services.

Pasted URLs are fetched only when they are public http(s) addresses. Local names, embedded passwords, and private, loopback, link-local, reserved, or multicast addresses are refused before any request. That check is not a promise against every SSRF trick. Keep the server bound to `127.0.0.1` unless you have your own network controls.

## Secrets

API keys belong in the environment or a local `.env` file, which is gitignored. Do not commit keys, tokens, or a real mailbox you do not want published. `.env.example` has placeholders only.

## Limits

Request bodies and paste length are capped. Outbound calls use a connect timeout, a read timeout, a small retry budget, and a per-host gap. A response larger than the configured cap is discarded. These limits reduce accidental load. They are not an authorization system.

No single source is allowed to decide a claim verdict on its own. Google Fact Check counts as one source. An outage is reported as unavailable, not as evidence that a citation is fabricated or that a claim is false.
