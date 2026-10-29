# Benchmark results

`make bench` writes a JSON file and a Markdown report into this directory. Both are gitignored. Commit neither the reports nor the downloaded datasets. The cache lives in `.cache/verifai-bench/`.

How to run a sample, how to read the metrics, and which label maps are in use is `docs/benchmark.md`.

A report is comparable to another report only when both name the same git SHA, model id, dataset sha256, seed, and limit. Live retrieval can still change a verdict between those runs. Offline passage fixtures freeze the pages. They do not score citations, because the citation checker calls live APIs.

CI does not run the benchmark. It runs the unit tests for the metric math and the loaders.
