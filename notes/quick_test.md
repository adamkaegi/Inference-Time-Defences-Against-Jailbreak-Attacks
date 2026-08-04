# Quick test

A fast smoke test that starts the app from a clean checkout and exercises
every registered attack and every registered defense once each. It uses
`prompts/quick_test.txt`, a batch with a single line, so the whole pass takes
a couple of minutes instead of the tens of minutes a full batch would need.

This only checks that each component runs end-to-end without raising — it is
not an attack-success or defense-effectiveness evaluation. Use
`python -m core.run_attack_defense_matrix` for that (see `README.md`).

## 1. Set up the environment

Run once per checkout, from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env`'s defaults are fine for this smoke test — no keys are required unless
you also plan to run the HarmBench/StrongREJECT judges (not used below; the
default `sample_safe_unsafe` judge needs no credentials).

## 2. Start Ollama and pull the models this test needs

Make sure the Ollama app/daemon is running (`ollama serve`, or launch the
desktop app), then pull the target and support models:

```bash
python scripts/pull_models.py
```

This pulls every model in `core/config.py`'s `TARGET_MODELS` and
`SUPPORT_MODELS`, including `qwen2.5:7b-instruct` (default target),
`dolphin-mistral:7b` (PAIR's attacker model), `llama-guard3:1b`, and
`llama3:8b`. The `perplexity` defense additionally downloads `gpt2` from
Hugging Face automatically on first use — no separate pull needed.

## 3. Dry-run sanity check (no Ollama needed)

```bash
python main.py --dry-run --batch quick_test
```

This should echo the prompt back with the selected attack/defense wrapping
applied and exit cleanly. If this fails, fix it before moving on — every
command below depends on the same wiring.

## 4. Run every attack once (defense off)

```bash
python main.py --batch quick_test --attack none --defense none
python main.py --batch quick_test --attack deepinception --defense none
python main.py --batch quick_test --attack gcg --defense none
python main.py --batch quick_test --attack pair --defense none
python main.py --batch quick_test --attack sample_hi_adam --defense none
```

`template` is a compatibility alias for `deepinception` (same instance), so
it isn't tested separately. `pair` is the slowest of the five — it runs
multiple refinement streams against the target model, so expect it to take
noticeably longer than the others even with one prompt.

## 5. Run every defense once (attack off)

```bash
python main.py --batch quick_test --attack none --defense none
python main.py --batch quick_test --attack none --defense smoothllm
python main.py --batch quick_test --attack none --defense self_reminder
python main.py --batch quick_test --attack none --defense perplexity
python main.py --batch quick_test --attack none --defense llama_guard_input
python main.py --batch quick_test --attack none --defense llama_guard_output
python main.py --batch quick_test --attack none --defense sample_bye_adam_input
python main.py --batch quick_test --attack none --defense sample_bye_adam_output
```

Defenses can also be stacked with a comma-separated list, e.g.
`--defense perplexity,smoothllm,llama_guard_output` (the proposal-aligned
stack used by the full matrix), if you want one extra command that checks
they compose correctly.

## 6. What to check

Each command above should:

- print one response line plus a judge label for the single prompt, and
- finish with `saved csv: outputs/run-quick_test-...csv`, with no traceback.

If a command raises instead, that attack/defense is broken — the traceback
will point at the failing module directly (`attacks/<name>.py` or
`defenses/<name>.py`).

## 7. Clean up (optional)

Every run above writes a CSV to `outputs/`. Since this is just a smoke test,
it's safe to delete them afterward:

```bash
rm outputs/run-quick_test-*.csv
```
