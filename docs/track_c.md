# Track C — Tasks & Evaluation (Yutong Dai)

Modules: `src/neuromesh/tasks/`, `src/neuromesh/evaluation/`, `src/neuromesh/baseline/`.
Tests: `tests/tasks/`, `tests/evaluation/`, `tests/baseline/`, fixtures in `tests/fixtures/transcripts/`.

## 1. Tasks

All three implement `contracts.Task` and are looked up with `neuromesh.tasks.get_task(name)`.
Instances are deterministic in `(task, seed, graph topology, n, graph seed)`. All content is
invented, so the model cannot answer from memory and every answer has an exact key.

| Task | Round-0 input | Agent `state` the engine must log | Readout |
| --- | --- | --- | --- |
| `fact_propagation` | Visual agents get 5 fictional facts, e.g. *"The access code of Project Kesoth-17 is 4821."* | `{"facts": {"F1": "4821", ...}}` | Frontoparietal agents |
| `consensus` | Every agent gets a private vote 0/1 (never unanimous at start) | `{"vote": 0}` | all agents |
| `aggregation` | Every agent of network *k* sees that network's secret integer (10–99, distinct) | `{"values": {"visual": 12, ...}, "sum": 345}` | Frontoparietal agents |

`TaskInstance.private_info` holds the round-0 text for each agent; `initial_state` the matching
state; `ground_truth` the answer key (plus `hop_distance` from the nearest Visual agent for fact
propagation, and `networks_present` for aggregation after node removal).

**Notes for Track B (prompts).** The state keys above are the whole interface: the scorer reads
only `CallRecord.state` (per round) and `RunResult.final_answers`. Values may be ints or numeric
strings. A call with `parse_ok = False` counts as an empty state.

## 2. Scores (`Task.score`, one dict per run)

* **Fact Propagation** — `readout_fidelity` (mean fraction of the 5 codes a readout agent holds
  exactly), `network_fidelity` (all agents), `readout_complete`, `distortion_rate` (wrong codes /
  reported codes), `propagation_delay_t50` (first round at which mean readout fidelity ≥ 0.5;
  censored at rounds + 1, with `t50_reached`), `fidelity_hop_<d>` and `fidelity_hop_slope`.
* **Consensus** — `agreement_ratio` (largest vote bloc / all agents; missing or invalid votes count
  as disagreeing), `consensus_reached`, `valid_vote_rate`, `convergence_round` (first unanimous
  round, censored), `initial_majority_share`.
* **Aggregation** — `exact_accuracy` (all 7 values and the sum right), `partial_accuracy` (fraction
  of the 7 values right), `values_complete` (all 7 values right, sum ignored: separates
  integration from arithmetic), `sum_accuracy`, `max_recoverable` (networks with a surviving agent / 7).

## 3. Evaluation (`neuromesh.evaluation`)

* `compute_metrics(runs, calls)` — one row per run. Task scores come from `Task.score` (one
  definition for tests and experiments); cost metrics come from the call log: `n_calls`,
  `prompt_tokens`, `completion_tokens`, `total_tokens`, `tokens_per_agent`, `max_context_tokens`
  (largest prompt of any call), `parse_error_rate`, `retries`, `wall_clock_s`.
* **Log format needed from the runner (Track D):** `runs` = the `RunResult` fields except `calls`,
  **plus an `instance` column holding the `TaskInstance` JSON** (it carries the answer key);
  `calls` = the `CallRecord` fields. `to_frames()` produces exactly this format.
* `summarize()` — mean, 95% t-interval, sd and count per condition (long table).
* `covariate_regression()` — OLS with HC1 robust SEs of a score on log2(N), topology dummies (ER
  reference) and Track A's structural covariates; constant/collinear columns are dropped and listed.
* `plots` — metric vs N per topology with CIs, fidelity vs hop distance, baseline vs network bars.
* `reference.run_reference()` — rule-based agents (no LLM) that follow the same synchronous protocol
  and log in the real format; used to validate the pipeline and as a communication-limited reference.
* `python -m neuromesh.evaluation.pilot --out results/pilot_reference` — runs the whole pipeline
  on all 25 graph conditions x 3 tasks x 5 seeds, ideal and 5%-noise relays (750 runs, ~15 s).

## 4. Single-agent baseline (`neuromesh.baseline`)

`run_baseline(instance, llm)` sends one prompt containing all task information to the same model
and scores the reply with `Task.score_baseline`. It never raises on a bad reply: transport errors
and unparseable answers are scored as failures and logged. Command line:

```bash
python -m neuromesh.baseline --base-url http://localhost:11434 --ollama --model qwen3:8b \
    --seeds 0 1 2 3 4 --n 14 --out results/baseline_n14
```

`--ollama` uses Ollama's native `/api/chat` with thinking disabled (Qwen3's OpenAI-compatible
route ignores `/no_think` and can spend the whole token budget on hidden reasoning). Without it,
any OpenAI-compatible endpoint works (vLLM, hosted APIs). The stand-in HTTP client in
`baseline/http_client.py` should be replaced by Track B's client once it lands.
