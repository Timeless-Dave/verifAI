# Accuracy benchmark

`make bench` scores VerifAI on labeled claims and writes the numbers a report has to show. It calls `run_check` in `src/verifai/orchestrate.py`. That is the same router, the same experts, and the same `corroborate` function the app uses. The benchmark does not contain a second verdict rule or a second judge.

CI runs the unit tests for the metrics and the loaders. It does not run `make bench`.

## What a number means

The scored classes are `supported`, `refuted`, and `unverified`.

The pipeline's `unsupported` verdict is refuted. `unverifiable` is unverified. `mixed` is scored as unverified, because the pipeline did not pick one side. A mixed row is an abstention. It is not a two-source block.

- **Accuracy** is correct labels divided by the sample.
- **Precision, recall, and F1** use the usual counts. Precision is n/a when that class was never predicted. Recall is n/a when the sample has no gold rows of that class. F1 is n/a when either denominator is empty. If both precision and recall are 0, F1 is 0.
- **Macro-F1** is the unweighted mean of the three F1 scores. An n/a F1 contributes 0. The per-class table still shows n/a, and the JSON keeps the counts.
- **Abstention rate** is how often the prediction is unverified.
- **Accuracy on non-abstained claims** ignores unverified predictions. It is n/a when the system never said supported or refuted.
- **Two-source block rate** is how often the verdict is unverifiable even though the retrieved evidence contains a supporting or refuting stance. `corroborate` withheld the side. A citation verdict is not a block: the citation checker does not use that rule, and one index can still verify a citation.
- **Mean latency** is the mean wall time of `run_check`, including retrieval.

The confusion matrix rows are gold and the columns are predicted, in the order supported, refuted, unverified.

## Datasets

Public files download into `.cache/verifai-bench/` and are gitignored.

| Dataset | Domain | Split | Why this split |
| --- | --- | --- | --- |
| FEVER v1.0 | general | `shared_task_dev.jsonl` | The blind test labels were not public. |
| SciFact v1.0 | health/science | `claims_dev.jsonl` | The test split has no gold labels. |
| LIAR v1.0 | news | `test.tsv` | Public labels, the split papers usually report. |
| Hand-written citations | citations | `src/verifai/bench/data/citations.jsonl` | Real works, invented citations, and two real works cited wrong on purpose. |

FEVER: `SUPPORTS` → supported, `REFUTES` → refuted, `NOT ENOUGH INFO` → unverified.

SciFact, one label per claim: only `SUPPORT` → supported, only `CONTRADICT` → refuted, no labeled evidence or both labels → unverified.

LIAR:

| LIAR label | Benchmark label |
| --- | --- |
| true, mostly-true | supported |
| false, pants-fire | refuted |
| half-true, barely-true | unverified |

Half-true and barely-true are mixed truth values. LIAR is research-use only. PolitiFact retains copyright in the statements. It is not an open-source license.

The router still decides which expert runs. A SciFact sentence with no health vocabulary can go to the general expert. A LIAR statement with no news vocabulary can go too. The report lists the experts that actually ran. Sentences shorter than 30 characters are not extracted as claims. If there is also no health or news cue, the row is unverifiable because nothing was checked.

When a citation paste is also extracted as a claim, `run_check` returns that claim's verdict as the top-level verdict. With no model loaded, the claim stays unverified even if the citation expert verified the reference. The JSON `experts` list shows both.

SciFact is not scored against its own abstract corpus or its gold cited documents. The health expert searches PubMed and Europe PMC. That number is not a SciFact leaderboard number.

## How a claim becomes supported

`corroborate` needs two independent sources that agree. Wikipedia is one source no matter how many articles were opened. Wikidata is a second source. The same paper in PubMed and Europe PMC is one source, because they share a DOI or PMID. Two different papers can agree. Google Fact Check is one source at most.

The `none` backend does not judge passages. Fact-check ratings that arrive with a stance are still read. With no such rating, every claim stays unverified, the block rate is 0, and accuracy equals the share of gold unverified rows.

## Running it on a 24 GB M5 Pro

The citation checker and the `none` backend do not load a model. 24 GB is far more than they need. The limit is the network.

```bash
cp .env.example .env
make bench BENCH_ARGS="--backend none --limit 50 --seed 0"
```

`--limit 50` keeps 50 rows from each dataset after a shuffle with `--seed`. `--limit 0` scores every row. The same seed, the same file hash, and the same limit pick the same ids. Live pages can still change the verdicts.

A full FEVER development file is about 20,000 claims. With live retrieval that is a long run. Stay with a limit until you mean to wait.

### NLI

The default model id is `MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli`, from `Settings`. It was trained on FEVER, ANLI, and other NLI sets. A FEVER score from this model is not an out-of-domain result. Say the model id next to the number. SciFact and LIAR are the more informative rows for this checkpoint.

The default environment does not install PyTorch, so CI stays small.

```bash
uv pip install torch transformers
make bench BENCH_ARGS="--backend nli --limit 50 --seed 0"
```

The small DeBERTa checkpoint is a few hundred megabytes and fits easily next to macOS.

### MLX, Ollama, llama.cpp

MLX is the Apple-silicon path. It is not installed in Linux CI.

```bash
uv pip install mlx-lm
make bench BENCH_ARGS="--backend mlx --model-id mlx-community/Qwen3.5-9B-4bit --limit 50 --seed 0"
```

The default MLX id in settings is that 4-bit 9B model. A 4-bit 9B is on the order of 6 GB of weights. That fits on 24 GB with room for the system. A 70B model does not.

```bash
make bench BENCH_ARGS="--backend ollama --model-id qwen3.5:9b --limit 50 --seed 0"
make bench BENCH_ARGS="--backend llamacpp --model-id qwen3.5-9b --limit 50 --seed 0"
```

Start the server yourself. Ollama listens on `127.0.0.1:11434`. llama.cpp's OpenAI-compatible server listens on `127.0.0.1:8080/v1`.

## Offline fixtures

`--offline` does not download and does not call the network. Claim files must already be in the cache, or you pass `--fever`, `--scifact`, and `--liar`. Passages come from `--passage-fixtures`, a JSON object whose keys are the claim texts:

```json
{
  "A clinical trial found this drug lowers blood pressure in patients with diabetes.": [
    {
      "source": "pubmed",
      "family": "pubmed",
      "title": "Trial one",
      "text": "Blood pressure fell.",
      "document_ids": ["pmid:1"],
      "native_stance": "supports"
    }
  ]
}
```

`native_stance` is optional. When it is set, the existing judge path treats it as a published rating and does not call a model. When it is omitted, the selected backend judges `text`. The experts and `corroborate` still run. Offline mode refuses citation rows.

## Output

Each run writes JSON and Markdown under `results/`. That directory is gitignored except for this folder's `README.md`. The files record the date, git SHA, VerifAI version, backend, model id, judge name, seed, limit, dataset version, sha256, and every scored example.

An example that raises is scored unverified and the error string is kept.
