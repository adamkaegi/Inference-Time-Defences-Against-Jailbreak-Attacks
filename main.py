"""Run a prompt (or a whole batch) through attack -> defense -> model -> defense -> judge.

Examples:
    python main.py "What is the capital of France?"
    python main.py                       # runs the batch set in config.py
    python main.py --batch instructions --defense sample_bye_adam_input,sample_bye_adam_output
    python main.py --dry-run             # no Ollama needed
"""

import argparse
import csv
import hashlib
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from attacks import ATTACKS
from core import config
from core.attack_cache import AttackOutputCache, settings_fingerprint
from core.console_io import configure_utf8_stdio
from core.evaluation_reporting import format_judge_digest
from core.pipeline import build_response_chain
from defenses import DEFENSES
from defenses.block import DefenseBlocked
from judges import JUDGES
from judges.runtime import (
    JudgeResult,
    evaluate_judge_batch,
    judge_requires_target_unload,
)
from prompts import available_batches, load_batch

RUN_FIELDNAMES = [
    "attack",
    "defenses",
    "judge",
    "judge_provider",
    "judge_model",
    "model",
    "batch",
    "prompt_set",
    "prompt_index",
    "input",
    "attacked_input",
    "output",
    "judge_label",
    "judge_score",
    "attack_latency_seconds",
    "response_pipeline_latency_seconds",
    "latency_seconds",
    "judge_batch_latency_seconds",
]


def _parse_defense_names(raw_value: str) -> list[str]:
    """argparse `type=` callback for --defense: "a,b" -> ["a", "b"], validated.

    Raising ArgumentTypeError here makes argparse print a clean usage error
    and exit, the same as an unknown --attack or --judge choice would.
    """
    names = [name.strip() for name in raw_value.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one defense name is required")
    unknown = [name for name in names if name not in DEFENSES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown defense(s): {', '.join(unknown)}")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", help="A single prompt. Omit to run a batch.")
    parser.add_argument(
        "--model",
        default=config.MODEL,
        help=f"Ollama target model (configured: {', '.join(config.TARGET_MODELS)}).",
    )
    parser.add_argument("--attack", default=config.ATTACK, choices=ATTACKS.keys())
    parser.add_argument(
        "--defense",
        default=_parse_defense_names(config.DEFENSE),
        type=_parse_defense_names,
        help="Comma-separated defense names.",
    )
    parser.add_argument("--judge", default=config.JUDGE, choices=JUDGES.keys())
    parser.add_argument("--batch", default=config.BATCH, choices=available_batches())
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Echo the target prompt; selected defenses still run.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Write checkpoints and the completed run to this exact CSV path.",
    )
    cache_options = parser.add_mutually_exclusive_group()
    cache_options.add_argument(
        "--attack-cache",
        type=Path,
        help=(
            "PAIR attack-output cache path. Matrix runs choose a shared path "
            "automatically."
        ),
    )
    cache_options.add_argument(
        "--no-attack-cache",
        action="store_true",
        help="Disable the automatic persistent PAIR cache.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible checkpoint supplied with --output-csv.",
    )
    args = parser.parse_args(argv)
    if args.resume and args.output_csv is None:
        parser.error("--resume requires --output-csv")
    return args


def _load_langfuse_handler() -> tuple[object | None, object | None]:
    """Return (client, callback_handler) for tracing chain runs in Langfuse.

    Langfuse is optional observability, not part of the pipeline itself, so
    if it's unconfigured or not installed we just skip tracing.
    """
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        return None, None

    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except ImportError:
        return None, None

    client = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        base_url=config.LANGFUSE_BASE_URL,
    )
    return client, CallbackHandler()


def _write_run_csv(
    run_rows: list[dict[str, str]],
    attack_name: str,
    defense_names: list[str],
    judge_name: str,
    model_name: str,
    batch_name: str | None,
    csv_path: Path | None = None,
) -> Path:
    if csv_path is None:
        outputs_dir = Path(__file__).resolve().parent / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = uuid4().hex[:8]
        csv_path = outputs_dir / f"run-{batch_name or 'single'}-{timestamp}-{run_id}.csv"

    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_name(
        f".{csv_path.name}.{uuid4().hex[:8]}.tmp"
    )
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=RUN_FIELDNAMES,
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in run_rows:
                writer.writerow(
                    {
                        **row,
                        # Canonical run metadata must not be overridden by a
                        # row loaded from an existing checkpoint.
                        "attack": attack_name,
                        "defenses": ",".join(defense_names),
                        "judge": judge_name,
                        "model": model_name,
                    }
                )
            csv_file.flush()
            os.fsync(csv_file.fileno())
        os.replace(temporary_path, csv_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return csv_path


def _load_resume_rows(
    csv_path: Path,
    prompts: list[str],
    attack_name: str,
    defense_names: list[str],
    judge_name: str,
    model_name: str,
    batch_name: str,
    prompt_set: str,
) -> list[dict[str, str]]:
    """Load and validate a checkpoint that is a contiguous prompt prefix."""
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing = set(RUN_FIELDNAMES).difference(fieldnames)
        if missing:
            raise ValueError(
                f"Resume checkpoint {csv_path} is missing column(s): "
                f"{', '.join(sorted(missing))}"
            )
        rows = [dict(row) for row in reader]

    if len(rows) > len(prompts):
        raise ValueError(
            f"Resume checkpoint {csv_path} has {len(rows)} rows for only "
            f"{len(prompts)} prompts"
        )

    expected_metadata = {
        "attack": attack_name,
        "defenses": ",".join(defense_names),
        "judge": judge_name,
        "model": model_name,
        "batch": batch_name,
        "prompt_set": prompt_set,
    }
    for index, row in enumerate(rows, 1):
        for field, expected in expected_metadata.items():
            if row.get(field, "") != expected:
                raise ValueError(
                    f"Resume checkpoint {csv_path} row {index} has "
                    f"{field}={row.get(field)!r}; expected {expected!r}"
                )
        if row.get("prompt_index", "") != str(index):
            raise ValueError(
                f"Resume checkpoint {csv_path} is not a contiguous prompt prefix: "
                f"row {index} has prompt_index={row.get('prompt_index')!r}"
            )
        if row.get("input", "") != prompts[index - 1]:
            raise ValueError(
                f"Resume checkpoint {csv_path} row {index} does not match "
                "the selected prompt batch"
            )
    return rows


def _unload_ollama_model(model_name: str) -> None:
    """Release target-model VRAM before a local GPU-backed judge loads."""
    result = subprocess.run(
        ["ollama", "stop", model_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        print(f"warning: could not unload Ollama model '{model_name}': {detail}")


def _configure_defense_for_run(
    defense: object,
    model_name: str,
    dry_run: bool,
    model_max_tokens: int,
) -> object:
    """Use an optional concrete run hook without expanding defenses/base.py."""
    for_run = getattr(defense, "for_run", None)
    if callable(for_run):
        if getattr(defense, "name", "") == "smoothllm":
            return for_run(
                model_name,
                dry_run=dry_run,
                max_tokens=model_max_tokens,
            )
        return for_run(model_name, dry_run=dry_run)
    return defense


def _configure_attack_for_run(attack: object, model_name: str, dry_run: bool) -> object:
    """Bind attacks with internal model calls to the selected target model."""
    for_run = getattr(attack, "for_run", None)
    if callable(for_run):
        return for_run(model_name, dry_run=dry_run)
    return attack


def _prompt_set_name(batch_name: str | None) -> str:
    normalized = (batch_name or "").lower()
    if "jailbreakbench_harmful" in normalized:
        return "harmful"
    if "jailbreakbench_benign" in normalized:
        return "benign"
    return "unknown"


def _safe_path_component(value: str) -> str:
    """Make model and batch names safe inside a cross-platform cache filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "value"


def _resolve_attack_cache(
    args: argparse.Namespace,
    attack: object,
    prompts: list[str],
    batch_name: str,
    csv_path: Path,
) -> AttackOutputCache | None:
    """Return the persistent PAIR cache shared by all matrix defense cells."""
    explicit_path = getattr(args, "attack_cache", None)
    disabled = bool(getattr(args, "no_attack_cache", False))
    attack_name = str(getattr(attack, "name", ""))
    if disabled or bool(getattr(args, "dry_run", False)):
        return None
    if attack_name != "pair":
        if explicit_path is not None:
            raise ValueError("--attack-cache is currently supported only for PAIR")
        return None

    identity_builder = getattr(attack, "cache_identity", None)
    if not callable(identity_builder):
        raise TypeError("PAIR must provide cache_identity() before caching its output")
    settings = dict(identity_builder())
    settings["prompt_batch_sha256"] = hashlib.sha256(
        "\n".join(prompts).encode("utf-8")
    ).hexdigest()
    fingerprint = settings_fingerprint(settings)[:12]

    if explicit_path is not None:
        cache_path = explicit_path.resolve()
    else:
        # Matrix checkpoints live under <output>/runs. Direct main.py runs use
        # an attack-cache directory beside their requested CSV.
        output_root = (
            csv_path.parent.parent
            if csv_path.parent.name.lower() == "runs"
            else csv_path.parent
        )
        cache_path = output_root / "attack-cache" / (
            f"pair-{_safe_path_component(str(getattr(args, 'model', 'model')))}-"
            f"{_safe_path_component(batch_name)}-{fingerprint}.json"
        )

    return AttackOutputCache(
        cache_path,
        attack=attack_name,
        model=str(getattr(args, "model", "")),
        batch=batch_name,
        settings=settings,
    )


def _apply_attack(
    attack: object,
    prompt: str,
    cache: AttackOutputCache | None,
) -> tuple[str, float, bool, float]:
    """Apply an attack or reuse it, returning wall time and cache provenance."""
    start_time = time.perf_counter()
    cached = cache.get(prompt) if cache is not None else None
    if cached is not None:
        return (
            cached.attacked_prompt,
            time.perf_counter() - start_time,
            True,
            cached.generation_latency_seconds,
        )

    attacked_prompt = attack.apply(prompt)
    generation_latency = time.perf_counter() - start_time
    if cache is not None:
        cache.put(prompt, attacked_prompt, generation_latency)
    return attacked_prompt, generation_latency, False, generation_latency


def _invoke_chain(chain, prompt: str, invoke_kwargs: dict) -> str:
    """Invoke the chain, returning a defense's response after an input block."""
    try:
        return chain.invoke(prompt, **invoke_kwargs)
    except DefenseBlocked as blocked:
        return blocked.response


def main() -> None:
    configure_utf8_stdio()
    args = parse_args()

    attack = _configure_attack_for_run(
        ATTACKS[args.attack], args.model, args.dry_run
    )
    model_max_tokens = config.model_max_tokens_for_attack(attack.name)
    defenses = [
        _configure_defense_for_run(
            DEFENSES[name],
            args.model,
            args.dry_run,
            model_max_tokens,
        )
        for name in args.defense
    ]
    judge = JUDGES[args.judge]

    # Apply the attack once in the loop so its exact output can be cached.
    # This chain handles the remaining defenses -> model -> defenses stages.
    response_chain = build_response_chain(
        defenses,
        args.model,
        dry_run=args.dry_run,
        max_tokens=model_max_tokens,
    )
    langfuse_client, langfuse_handler = _load_langfuse_handler()

    single_prompt = args.prompt is not None
    prompts = [args.prompt] if single_prompt else load_batch(args.batch)
    source = "single prompt" if single_prompt else f"batch '{args.batch}' ({len(prompts)} prompts)"
    batch_name = "single" if single_prompt else args.batch
    prompt_set = _prompt_set_name(None if single_prompt else args.batch)
    judge_provider = str(getattr(judge, "provider", ""))
    judge_model = str(getattr(judge, "model_name", ""))

    defense_summary = ", ".join(f"{defense.name}:{defense.stage}" for defense in defenses)
    print(f"attack={attack.name}  defenses=[{defense_summary}]  judge={judge.name}  model={args.model}")
    print(f"running: {source}\n")

    requested_csv = args.output_csv.resolve() if args.output_csv is not None else None
    if requested_csv is not None and requested_csv.exists() and not args.resume:
        raise FileExistsError(
            f"Output CSV already exists; pass --resume to validate and continue it: "
            f"{requested_csv}"
        )

    if args.resume and requested_csv is not None and requested_csv.exists():
        run_rows = _load_resume_rows(
            requested_csv,
            prompts,
            attack.name,
            args.defense,
            judge.name,
            args.model,
            batch_name,
            prompt_set,
        )
        csv_path = requested_csv
        print(f"resuming {len(run_rows)}/{len(prompts)} cached response(s)")
    else:
        run_rows: list[dict[str, str]] = []
        csv_path = _write_run_csv(
            run_rows,
            attack.name,
            args.defense,
            judge.name,
            args.model,
            batch_name=None if single_prompt else args.batch,
            csv_path=requested_csv,
        )
    print(f"checkpoint csv: {csv_path}")

    attack_cache = _resolve_attack_cache(
        args,
        attack,
        prompts,
        batch_name,
        csv_path,
    )
    if attack_cache is not None:
        # Backfill a cache when resuming a PAIR checkpoint created before the
        # cache integration. A conflicting value fails closed.
        for cached_row in run_rows:
            attack_cache.put(
                cached_row["input"],
                cached_row["attacked_input"],
                float(cached_row.get("attack_latency_seconds", 0.0) or 0.0),
            )
        print(f"PAIR attack cache: {attack_cache.path}")

    for i in range(len(run_rows) + 1, len(prompts) + 1):
        prompt = prompts[i - 1]
        print(f"--- [{i}] {prompt}")
        invoke_kwargs = (
            {
                "config": {
                    "callbacks": [langfuse_handler],
                    "run_name": f"[{i}] {prompt[:80]}",
                }
            }
            if langfuse_handler
            else {}
        )
        (
            attacked_prompt,
            attack_latency_seconds,
            attack_cache_hit,
            original_attack_latency_seconds,
        ) = _apply_attack(attack, prompt, attack_cache)
        if attack_cache_hit:
            print(
                "PAIR attack cache hit "
                f"(original generation {original_attack_latency_seconds:.3f}s)"
            )

        response_start_time = time.perf_counter()
        output_text = _invoke_chain(response_chain, attacked_prompt, invoke_kwargs)
        response_latency_seconds = time.perf_counter() - response_start_time
        latency_seconds = attack_latency_seconds + response_latency_seconds
        row = {
            "judge_provider": judge_provider,
            "judge_model": judge_model,
            "batch": batch_name,
            "prompt_set": prompt_set,
            "prompt_index": str(i),
            "input": prompt,
            "attacked_input": attacked_prompt,
            "output": output_text,
            "judge_label": "",
            "judge_score": "",
            "attack_latency_seconds": f"{attack_latency_seconds:.3f}",
            "response_pipeline_latency_seconds": f"{response_latency_seconds:.3f}",
            "latency_seconds": f"{latency_seconds:.3f}",
            "judge_batch_latency_seconds": "",
        }
        run_rows.append(row)
        _write_run_csv(
            run_rows,
            attack.name,
            args.defense,
            judge.name,
            args.model,
            batch_name=None if single_prompt else args.batch,
            csv_path=csv_path,
        )
        print(output_text)
        print(
            f"attack_latency={attack_latency_seconds:.3f}s  "
            f"response_latency={response_latency_seconds:.3f}s  "
            f"total_latency={latency_seconds:.3f}s\n"
        )

    # Cache target-model outputs before invoking a potentially expensive judge.
    # If judge setup fails (for example, gated weights are unavailable), the
    # generated responses are still available for a later offline judge pass.
    if args.dry_run:
        judge_results = [JudgeResult(label="Skipped (dry-run)") for _ in run_rows]
        judge_latency_seconds = 0.0
    else:
        if judge_requires_target_unload(judge):
            print(f"unloading Ollama target {args.model} before local judge...")
            _unload_ollama_model(args.model)
        print(f"judging {len(run_rows)} response(s) with {judge.name}...")
        judge_start_time = time.perf_counter()
        try:
            judge_results = evaluate_judge_batch(
                judge,
                [row["input"] for row in run_rows],
                [row["output"] for row in run_rows],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Judge '{judge.name}' failed; responses remain cached at {csv_path}"
            ) from exc
        judge_latency_seconds = time.perf_counter() - judge_start_time

    if len(judge_results) != len(run_rows):
        raise RuntimeError(
            f"Judge '{judge.name}' returned {len(judge_results)} result(s) for "
            f"{len(run_rows)} response(s); responses remain cached at {csv_path}"
        )

    for index, (row, result) in enumerate(zip(run_rows, judge_results), 1):
        # A resumed run is judged as one batch; provenance must describe the
        # labels just written, not the process that created cached responses.
        row["judge_provider"] = judge_provider
        row["judge_model"] = judge_model
        row["judge_label"] = result.label or ""
        row["judge_score"] = "" if result.score is None else f"{result.score:.6f}"
        row["judge_batch_latency_seconds"] = f"{judge_latency_seconds:.3f}"
        print(f"judge [{index}] {result.display()}")

    print()
    for line in format_judge_digest(judge.name, judge_results):
        print(line)

    _write_run_csv(
        run_rows,
        attack.name,
        args.defense,
        judge.name,
        args.model,
        batch_name=None if single_prompt else args.batch,
        csv_path=csv_path,
    )
    print(f"judge batch latency: {judge_latency_seconds:.3f}s")
    print(f"saved csv: {csv_path}")

    if langfuse_client is not None:
        langfuse_client.flush()


if __name__ == "__main__":
    main()
