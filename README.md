# Jailbreak — minimal attack/defense chain

A tiny LangChain harness that runs a prompt through a swappable
**attack**, a swappable **defense**, a swappable **judge**, and a local Ollama model.

## Flow

```
original prompt -> attack -> [input defenses] -> model -> [output defenses] -> response
       |                                                                  |
       +---------------------------- judge -------------------------------+
```

Each defense declares a `stage` (`"input"` or `"output"`), and the chain plugs
every defense in at the right point automatically. The judge runs after the
chain and leaves the response text unchanged. Pairwise judges receive the
original behavior plus the final response; response-only demo judges remain
compatible. Multiple defenses are applied in the order you list them, grouped
by stage.

## Layout

```
attacks/     one file per attack + context-only base.py + registry in __init__.py
defenses/    one file per defense + context-only base.py + registry in __init__.py
judges/      one file per judge + context-only base.py + runtime compatibility
prompts/     one .txt per batch (one prompt per line) + loader
core/        shared internals: config.py, pipeline.py, attack_cache.py,
             console_io.py, evaluation_reporting.py, run_attack_defense_matrix.py
main.py      CLI entry point
outputs/     CSV files written per run
scripts/     shell scripts that run main.py multiple times consecutively
```

## Sample components

- **Attack** `sample_hi_adam` — appends `Start all messages with "Hi Adam".`
- **Defense** `sample_bye_adam_input` (input) — appends `End all messages with "Bye Adam".` to the prompt
- **Defense** `sample_bye_adam_output` (output) — appends `Bye Adam` to the model's response
- **Judge** `sample_safe_unsafe` — labels the model output as `Safe` or `Unsafe` with a simple keyword check

## Attacks

- **Attack** `deepinception` — Ryan's five-layer nested-scene template attack
- **Attack** `template` — compatibility alias for `deepinception`
- **Attack** `gcg` — appends a precomputed, static GCG-style universal adversarial suffix to the prompt
- **Attack** `pair` — deterministically refines the prompt in a black-box loop
  using seeded attacker/target decoding and a refusal-first lightweight stopping
  oracle (up to five rounds by default). Matrix runs persist each selected PAIR
  prompt once and reuse that exact prompt across every defense cell; the cache
  identity includes the PAIR settings, target model, and prompt-batch hash
- **Attack** `none` — raw-prompt baseline
- **Attack** `sample_hi_adam` — original wiring example

## Defences

- **Defense** `smoothllm` (generation) — generates five Bernoulli
  non-whitespace character-replacement perturbations (`q=10%`), obtains five
  target responses, and returns a response from the majority
  refusal/non-refusal class; this is a documented randomized-smoothing variant
  and does not make an extra target call after voting
- **Defense** `self_reminder` (input) — wraps the prompt with a responsible-assistant prefix and a safety-review suffix; both are overridable on the defense constructor
- **Defense** `perplexity` (input) — scores the attacked prompt with local GPT-2 and blocks scores above the configured threshold
- **Defense** `llama_guard_input` (input) — classifies the model-facing prompt with Llama Guard and skips target inference when unsafe
- **Defense** `llama_guard_output` (output) — classifies and replaces unsafe model output

## Judges

- **Judge** `sample_safe_unsafe` — keyword-only wiring/debug judge; its terminal
  summary is explicitly marked **not final ASR** and must not be reported
- **Judge** `strongreject` — local fine-tuned StrongREJECT evaluator; produces
  a continuous harmful-assistance score from 0 (refusal/useless) to 1
  (maximally useful harmful assistance)
- **Judge** `harmbench_mistral_7b_val_cls` — official binary HarmBench Mistral
  validation classifier (`Unsafe`/1 for Yes, `Safe`/0 for No)
- **Judge** `jbb_refusal_llama3_8b` — JailbreakBench Llama 3 8B refusal
  classifier (`Refused`/1 or `Not Refused`/0); the old
  `jailbreak_bench_llama8b` name remains as a CLI alias

The Mistral checkpoint is explicitly HarmBench's validation classifier. It is
available as this harness's post-generation judge as requested, but HarmBench's
canonical test/final classifier is the separate Llama-2 13B classifier.

## Prompt batches

- `general` — everyday questions
- `instructions` — instruction-following prompts
- `jailbreakbench_harmful_100` / `_10` — pinned official JBB misuse goals
- `jailbreakbench_benign_100` / `_10` — pinned thematically matched benign goals

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-cuda.txt
ollama serve # separate terminal
python scripts\pull_models.py
python scripts\prefetch_strongreject.py
python scripts\prefetch_harmbench.py
python scripts\fetch_jailbreakbench_prompts.py
ollama pull llama3:8b # local JBB refusal provider
```

The two evaluator prefetch commands download large Hugging Face checkpoints;
HarmBench is approximately 14.5 GB. StrongREJECT also requires accepting the
`google/gemma-2b` license and setting `HF_TOKEN` in `.env`.

The JBB refusal judge defaults to the installed Ollama model. For the exact
Together route used by JailbreakBench, set these in `.env`:

```dotenv
JBB_REFUSAL_PROVIDER=together
TOGETHER_API_KEY=...
```

## Local Langfuse

If you want to trace runs locally, start a Langfuse instance in a separate terminal or repo first:

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up -d
```

Then create a Langfuse project in the UI at http://localhost:3000 and put its
keys in a local `.env` copied from `.env.example`:

```bash
LANGFUSE_PUBLIC_KEY = "..."
LANGFUSE_SECRET_KEY = "..."
LANGFUSE_BASE_URL = "http://localhost:3000"
```


## Flags

The CLI accepts these flags:

- `--model` sets the Ollama model tag to use. Default: `qwen2.5:7b-instruct`.
- `--attack` selects the attack implementation from `attacks/__init__.py`. Default: `deepinception`.
- `--defense` selects one or more defenses from `defenses/__init__.py` as a comma-separated list. Default: `none`.
- `--judge` selects the judge implementation from `judges/__init__.py`. Default: `sample_safe_unsafe`.
- `--batch` chooses the prompt batch file stem from `prompts/`. Default: `general`.
- `--dry-run` skips Ollama and echoes the prompt flow for wiring checks.

A `prompt` is optional. If you omit it, the selected batch runs instead.

Defaults live in `core/config.py`; every one is overridable with the flags above.

## Run

```bash
python main.py "What is the capital of France?"       # single prompt
python main.py                                        # batch from core/config.py
python main.py --batch instructions --defense sample_bye_adam_input,sample_bye_adam_output
python main.py --judge sample_safe_unsafe             # lightweight debug judge
python main.py --judge strongreject                   # continuous harmful score
python main.py --judge harmbench_mistral_7b_val_cls   # binary validation classifier
python main.py --judge jbb_refusal_llama3_8b          # refusal classification
python main.py --dry-run                              # no Ollama, tests wiring
```

## Final evaluation matrix

Refusal and harmful-behavior success are deliberately reported separately:

| Prompt set | Judge | Final metric |
| --- | --- | --- |
| Harmful JBB | HarmBench Mistral | Behavior-level attack success rate |
| Harmful JBB | JBB refusal judge | No-refusal rate |
| Benign JBB | JBB refusal judge | False-refusal / over-blocking rate |

A non-refusal is not automatically a successful attack. The report therefore
also counts responses that did not refuse but still failed HarmBench.

Run the complete attack/defense experiment, including benign no-attack
usability cells, with:

```powershell
.venv\Scripts\python.exe -m core.run_attack_defense_matrix
```

The default experiment is 30 cells: four attacks × six defense conditions on
harmful prompts, plus six benign no-attack usability cells. The sixth
condition is the proposal-aligned input → generation → output stack
`perplexity,smoothllm,llama_guard_output`.

Ordinary target generations use a 512-token limit. DeepInception and its
`template` alias use 2048 tokens because the nested-scene response needs more
room to reach its final layer. The same per-attack limit applies to SmoothLLM's
internal target generations.

The no-flag default uses a paired 50-harmful/50-benign midterm set sampled from
alternating indices of the full dataset. Use `--quick` for a 10/10 smoke test or
`--full` for all 100/100 prompts. Every cell writes directly to a deterministic
UTF-8 checkpoint, and the manifest is updated after each cell.
If a run is interrupted, resume it without regenerating completed responses:

```powershell
.venv\Scripts\python.exe -m core.run_attack_defense_matrix `
  --quick `
  --resume `
  --output-dir outputs\matrix-YYYYMMDD-HHMMSS-ID
```

The resume command must use the same model, provider, quick/full setting,
attacks, defenses, and judges. It validates row metadata, settings, prompt
hashes, and the exact cached prompt prefix before continuing. Checkpoints that
do not have a schema-v2 manifest are not reused because their decoding and
defense settings cannot be proven; they remain untouched while new `-v2.csv`
checkpoints are produced. The final evaluator also resumes its authoritative
`evaluated_rows.csv`, so a judge failure does not discard completed labels.

To evaluate already cached run CSVs without regenerating target responses:

```powershell
.venv\Scripts\python.exe scripts\evaluate_matrix.py `
  outputs\run-jailbreakbench_harmful_100-....csv `
  outputs\run-jailbreakbench_benign_100-....csv
```

Add `--resume --output-dir PATH` to that evaluator command to validate and
continue a partial evaluation report. Add `--force` only when deliberately
re-evaluating enabled judge labels with the currently configured provenance.

Each evaluation directory contains:

- `manifest.json`: prompt hashes, target/PAIR settings, final-judge provider,
  code revision/dirty status, configured failure policies, deterministic cell
  paths, and live progress.
- `evaluated_rows.csv`: every prompt/response, both labels, scores, provenance,
  timings, and the full 2x2 behavior-success/refusal result.
- `experiment_matrix.csv`: one statistics row per observed model × attack ×
  defense × prompt-set run, including duplicate/completion checks, latency
  versus the matching no-defense baseline, and stack-versus-best-single ASR.
- `summary.md`: a numbered presentation-friendly digest; the same list is
  printed in the terminal. It includes latency and defense-stack comparisons.

Perplexity and Llama Guard use the `raise` failure policy by default for
research runs. If either defense cannot load or classify, the cell stops with
its completed rows checkpointed instead of silently counting a fail-open path
as a valid observation.

## Adding real components later

- New attack: add a file in `attacks/`, subclass `Attack`, register it in `attacks/__init__.py`.
- New defense: add a file in `defenses/`, subclass `Defense`, set `stage`, register it in `defenses/__init__.py`.
- New judge: add a file in `judges/`, subclass `Judge`, register it in `judges/__init__.py`.
- New batch: drop a `.txt` file in `prompts/` — it's auto-discovered by its filename.

Nothing in `core/pipeline.py` or `main.py` changes when you do.

## Scripts

- `python -m core.run_attack_defense_matrix` — runs the full research attack × defense
  matrix on harmful prompts plus no-attack benign usability cells, caches each
  response once, and then uses HarmBench plus the JBB refusal judge. Repeat
  `--judge` to select a subset explicitly. Use
  `--quick` for paired 10-prompt smoke batches. Repeat `--defense` to select
  matrix entries; comma-separated defense names form an ordered stack. Use
  `--resume --output-dir PATH` to continue a validated checkpoint.
- `scripts/pull_models.py` — pulls every configured Ollama target/support model.
- `scripts/prefetch_strongreject.py` — downloads and smoke-tests the pinned
  StrongREJECT adapter and Gemma base.
- `scripts/prefetch_harmbench.py` — downloads and smoke-tests the pinned
  HarmBench Mistral validation classifier.
- `scripts/judge_csv.py` — applies a judge to cached responses without
  regenerating target-model output; it always writes a new CSV.
- `scripts/evaluate_matrix.py` — adds the selected final labels to cached runs
  and writes the row-level CSV, aggregate matrix CSV, and digest.
- `scripts/fetch_jailbreakbench_prompts.py` — recreates the 100- and 10-goal
  harmful/benign batches from a pinned official JBB dataset revision.
- `scripts/run_readme_examples.sh` — runs the lightweight README examples.
- `scripts/script.sh` — runs the attack/defense combinations.

## Tests

Run the complete suite with:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use fake evaluators/models for heavyweight paths. They check attack
formatting, registry/default selection, pairwise judge wiring, score parsing,
model revision pins, deterministic target settings, one-time attack execution,
atomic/resumable UTF-8 checkpoints, matrix progress manifests, generation-stage
response voting, CSV re-judging behavior, and Ollama VRAM unloading. Separate
prefetch scripts provide the real checkpoint smoke tests.
