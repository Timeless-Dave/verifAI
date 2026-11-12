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
- **Blocked rows whose evidence leaned the gold way** looks only at two-source blocks. It takes the side most judged passages took (ties have no side) and compares it with the gold label. It is a diagnostic. It never changes a prediction, and it shows what relaxing the two-source rule would gain or cost.
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

Reference entries, bare DOIs, and lines that are only a link go to the citation expert alone. They are not split into sentences and sent to the claim experts. A paste that mixes prose with references still has its prose sentences checked as claims. When that happens, `run_check` returns the claims' verdict as the top-level verdict, and the JSON `experts` list shows the citation expert too.

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

The default model id is `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, from `Settings`. It was trained on FEVER, ANLI, and other NLI sets. A FEVER score from this model is not an out-of-domain result. Say the model id next to the number. SciFact and LIAR are the more informative rows for this checkpoint.

The default environment does not install PyTorch, so CI stays small. `EXTRAS="mlx nli"` adds the `nli` extra (PyTorch and transformers). On Linux use `EXTRAS=nli`.

```bash
make bench EXTRAS="mlx nli" BENCH_ARGS="--backend nli --limit 50 --seed 0"
```

The DeBERTa base checkpoint is about 370 MB and fits easily next to macOS. An earlier default, `MoritzLaurer/DeBERTa-v3-small-mnli-fever-anli`, is not a public model on Hugging Face and could not be downloaded.

### MLX, Ollama, llama.cpp

MLX is the Apple-silicon path. `make` installs the `mlx` extra there by default. It is not installed in Linux CI.

```bash
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

## Results for v0.2.0

Run on a 24 GB Apple silicon Mac: `make bench BENCH_ARGS="--backend mlx --limit 50 --seed 0"`, model `mlx-community/Qwen3.5-9B-4bit`, git `370ce90`, 2026-11-07. Live retrieval, no Google Fact Check key. 50 rows each from FEVER, SciFact, and LIAR, plus the 16 hand-written citations. The JSON and Markdown reports are attached to the GitHub release.

| Domain | n | Accuracy | Abstention | Correct when it answered | Two-source blocks |
| --- | ---: | ---: | ---: | ---: | ---: |
| general (FEVER) | 50 | 0.38 | 1.00 | n/a (0 answered) | 0.64 |
| health (SciFact) | 50 | 0.40 | 0.82 | 4/9 | 0.16 |
| news (LIAR) | 50 | 0.52 | 1.00 | n/a (0 answered) | 0.32 |
| citations | 16 | 1.00 | 0.13 | 14/14 | 0 |
| overall | 166 | 0.49 | 0.86 | 18/23 | 0.34 |

Mean latency was 5.3 s per check. Overall macro-F1 was 0.40.

What these numbers say:

- The citation checker is the part that works. It found every invented reference and every real one.
- Claim checking abstains on most claims. On general and news claims it never gave a verdict in this sample. The accuracy in those rows is the share of gold rows that are unverified.
- Health verdicts that passed the two-source rule were right 4 of 9 times. Two papers that a 9B model reads as agreeing are not yet enough.

### Why the two-source rule stays

A blocked claim is one where the judged evidence took a side but no second independent source agreed. In this run, the side the evidence leaned matched the gold label in 20 of 53 blocked claims (0.38): 18/31 on FEVER, 1/8 on SciFact, and 1/14 on LIAR. Releasing those verdicts would give answers that are wrong more often than right, and on FEVER, where it does best, wrong 42% of the time. VerifAI keeps the rule and says unverified instead.

Two fixes made before this run matter for reading older numbers. Qwen3.5 thinks before it answers by default. Its reasoning filled the token budget, so no passage was judged at all, and every FEVER and SciFact claim abstained. Thinking is now off for MLX, Ollama, and llama.cpp. The judge also read a passage that left a detail out as refuting the claim. The prompt now says missing information is not a contradiction. Blocks fell from 0.48 to 0.34, and refute leans from 58 to 42.

The way to more verdicts is more independent sources for general and news claims, not a looser rule. A Google Fact Check key adds PolitiFact and other reviews, which is the obvious next step for LIAR-style statements.

## Results after v0.2.0 (unreleased)

Same command, sample, and model as the v0.2.0 table, run on 2026-11-12 with the fixes since then: claims are only real sentences, Wikidata gives the judge each item's key facts, a clear majority of one source's articles counts once, and fact-check reviews are judged against the claim. The first run had no Google Fact Check key (code at `be8618b`; its report wrongly names `7fbfedb`, a bug fixed in `fb5961c`). The second ran at `fb5961c` with a key.

| | v0.2.0 | No fact-check key | With fact-check key |
| --- | ---: | ---: | ---: |
| Accuracy | 0.488 | 0.518 | 0.518 |
| Abstention | 0.861 | 0.831 | 0.825 |
| Correct when it answered | 18/23 | 23/28 | 23/29 |
| General (FEVER) answered | 0 | 7 (4 right) | 7 (4 right) |
| Health (SciFact) answered | 9 (4 right) | 7 (5 right) | 7 (5 right) |
| News (LIAR) answered | 0 | 0 | 1 (0 right) |
| Two-source blocks | 0.337 | 0.259 | 0.253 |
| Citations | 16/16 | 16/16 | 16/16 |
| Mean latency | 5.3 s | 8.3 s | 8.8 s |

Of the wrong FEVER verdicts, two disagree with the label more than with the world. FEVER marks "Rupert Murdoch is a business owner" and "Konidela Production Company was established by the son of Shah Rukh Khan" as not enough information. VerifAI supported the first and refuted the second (the company was founded by Ram Charan). "Maria Theresa ruled for 40 years" is a real error: she ruled from 1740 to 1780.

A Google Fact Check key barely moves LIAR. Its statements are from 2007 to 2016, a matching review is rare, and a review is one source that still needs a second. The key helps most with recent viral claims, which this sample does not measure. The one LIAR verdict it enabled was wrong: PolitiFact rated the Syria chemical-weapons statement Mostly True, and VerifAI refuted it.

Latency rose by about 3 s per check, because Wikidata now takes up to five requests a claim.
