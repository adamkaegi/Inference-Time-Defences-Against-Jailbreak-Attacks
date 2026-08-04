"""Run the research attack/defense matrix and evaluate every cached response.

Examples (run from the project root):
    python -m core.run_attack_defense_matrix
    python -m core.run_attack_defense_matrix --quick
    python -m core.run_attack_defense_matrix --model dolphin-mistral:7b
    python -m core.run_attack_defense_matrix --defense self_reminder,perplexity,llama_guard_output
    python -m core.run_attack_defense_matrix --defense none --defense self_reminder,perplexity

Each repeated ``--defense`` value is one matrix column.  Names within a value
are a stack and are passed to ``main.py`` in the given order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from attacks import ATTACKS
from core import config
from core.console_io import configure_utf8_stdio
from defenses import DEFENSES
from main import _load_resume_rows
from prompts import load_batch
from scripts.evaluate_matrix import evaluate_matrix
from scripts.fetch_jailbreakbench_prompts import JBB_DATA_REVISION


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ATTACKS = ("none", "deepinception", "gcg", "pair")
DEFAULT_DEFENSE_STACKS = (
    "none",
    "self_reminder",
    "smoothllm",
    "perplexity",
    "llama_guard_output",
    "perplexity,smoothllm,llama_guard_output",
)
DEFAULT_JUDGES = ("harmbench_mistral", "jailbreak_bench_llama8b")
JUDGE_CHOICES = (
    "harmbench_mistral",
    "jailbreak_bench_llama8b",
    "jbb_refusal_llama3_8b",
)
FULL_HARMFUL_BATCH = "jailbreakbench_harmful_100"
MIDTERM_HARMFUL_BATCH = "jailbreakbench_harmful_50"
QUICK_HARMFUL_BATCH = "jailbreakbench_harmful_10"
FULL_BENIGN_BATCH = "jailbreakbench_benign_100"
MIDTERM_BENIGN_BATCH = "jailbreakbench_benign_50"
QUICK_BENIGN_BATCH = "jailbreakbench_benign_10"


def _attack_name(value: str) -> str:
    if value not in ATTACKS:
        raise argparse.ArgumentTypeError(f"unknown attack: {value}")
    return value


def _defense_stack(value: str) -> str:
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise argparse.ArgumentTypeError("a defense stack cannot be empty")
    unknown = [name for name in names if name not in DEFENSES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown defense(s): {', '.join(unknown)}"
        )
    if "none" in names and len(names) > 1:
        raise argparse.ArgumentTypeError(
            "'none' must be used alone, not inside a defense stack"
        )
    return ",".join(names)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every selected attack against every selected defense stack, "
            "then judge the cached responses."
        )
    )
    parser.add_argument(
        "--model",
        default=config.MODEL,
        help=f"Ollama target model (default: {config.MODEL}).",
    )
    parser.add_argument(
        "--attack",
        action="append",
        type=_attack_name,
        dest="attacks",
        help="Attack to include; repeat to choose a subset.",
    )
    parser.add_argument(
        "--defense",
        action="append",
        type=_defense_stack,
        dest="defense_stacks",
        metavar="NAME[,NAME...]",
        help=(
            "One defense matrix entry; comma-separated names form a stack. "
            "Repeat this option for multiple entries."
        ),
    )
    parser.add_argument(
        "--judge",
        action="append",
        choices=JUDGE_CHOICES,
        dest="judges",
        help="Final response judge; repeat to choose a subset (default: both).",
    )
    parser.add_argument(
        "--jbb-provider",
        choices=("ollama", "together"),
        default=config.JBB_REFUSAL_PROVIDER,
        help="Provider for the JailbreakBench Llama-8B judge.",
    )
    prompt_size = parser.add_mutually_exclusive_group()
    prompt_size.add_argument(
        "--quick",
        action="store_true",
        help="Use the paired 10-prompt harmful and benign smoke-test batches.",
    )
    prompt_size.add_argument(
        "--full",
        action="store_true",
        help="Use all 100 harmful and 100 benign prompts instead of the default 50/50 midterm set.",
    )
    parser.add_argument(
        "--skip-benign",
        action="store_true",
        help="Skip benign no-attack usability cells and over-blocking metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New result directory, or an existing checkpoint with --resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and continue an existing --output-dir checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise matrix wiring without target inference or final judges.",
    )
    args = parser.parse_args(argv)
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    args.attacks = args.attacks or list(DEFAULT_ATTACKS)
    args.defense_stacks = args.defense_stacks or list(DEFAULT_DEFENSE_STACKS)
    args.judges = args.judges or list(DEFAULT_JUDGES)
    return args


def _matrix_cells(args: argparse.Namespace) -> list[tuple[str, str, str, str]]:
    if args.quick:
        harmful_batch = QUICK_HARMFUL_BATCH
        benign_batch = QUICK_BENIGN_BATCH
    elif args.full:
        harmful_batch = FULL_HARMFUL_BATCH
        benign_batch = FULL_BENIGN_BATCH
    else:
        harmful_batch = MIDTERM_HARMFUL_BATCH
        benign_batch = MIDTERM_BENIGN_BATCH
    cells = [
        ("harmful", harmful_batch, attack, defense_stack)
        for attack in args.attacks
        for defense_stack in args.defense_stacks
    ]
    if not args.skip_benign:
        cells.extend(
            ("benign", benign_batch, "none", defense_stack)
            for defense_stack in args.defense_stacks
        )
    return cells


def _child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONIOENCODING": "utf-8:backslashreplace",
            "PYTHONUTF8": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _run_and_find_csv(command: list[str], expected_path: Path | None = None) -> Path:
    """Stream a main.py run and return the path printed by its final checkpoint."""
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=_child_environment(),
    )
    saved_csv: Path | None = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        if line.startswith("saved csv: "):
            saved_csv = Path(line.removeprefix("saved csv: ").strip())

    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    if saved_csv is None:
        raise RuntimeError("main.py completed without reporting its output CSV")
    if not saved_csv.is_absolute():
        saved_csv = PROJECT_ROOT / saved_csv
    saved_csv = saved_csv.resolve()
    if expected_path is not None and saved_csv != expected_path.resolve():
        raise RuntimeError(
            f"main.py saved {saved_csv}, expected checkpoint {expected_path.resolve()}"
        )
    return saved_csv


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "outputs" / f"matrix-{stamp}-{uuid4().hex[:8]}"


def _cell_name(
    index: int,
    prompt_set: str,
    attack: str,
    defense_stack: str,
) -> str:
    return (
        f"{index:03d}-{prompt_set}-{attack}-"
        f"{defense_stack.replace(',', '+')}-v2.csv"
    )


def _matching_checkpoint(
    preferred_path: Path,
    *,
    allow_existing: bool,
) -> Path:
    if preferred_path.exists():
        if not allow_existing:
            raise ValueError(
                f"Cannot reuse {preferred_path} without a compatible schema-v2 "
                "manifest proving its run settings"
            )
        return preferred_path
    return preferred_path


def _validated_checkpoint_rows(
    path: Path,
    *,
    model: str,
    batch: str,
    prompt_set: str,
    attack: str,
    defenses: str,
) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return _load_resume_rows(
        path,
        load_batch(batch),
        attack,
        defenses.split(","),
        "sample_safe_unsafe",
        model,
        batch,
        prompt_set,
    )


def _batch_provenance(batch: str) -> dict[str, object]:
    prompts = load_batch(batch)
    content = "\n".join(prompts).encode("utf-8")
    return {
        "prompt_count": len(prompts),
        "sha256": hashlib.sha256(content).hexdigest(),
        "jailbreakbench_revision": JBB_DATA_REVISION,
    }


def _git_provenance() -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    manifest["updated_at"] = datetime.now().astimezone().isoformat()
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex[:8]}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _defense_settings() -> dict[str, object]:
    return {
        "smoothllm_num_samples": config.SMOOTHLLM_NUM_SAMPLES,
        "smoothllm_perturbation_rate": config.SMOOTHLLM_PERTURBATION_RATE,
        "perplexity_threshold": config.PERPLEXITY_THRESHOLD,
        "perplexity_failure_policy": config.PERPLEXITY_FAILURE_POLICY,
        "llama_guard_failure_policy": config.LLAMA_GUARD_FAILURE_POLICY,
    }


def _attack_settings() -> dict[str, object]:
    return {
        "deepinception_scene": getattr(ATTACKS.get("deepinception"), "scene", ""),
        "deepinception_layers": getattr(
            ATTACKS.get("deepinception"), "layer_count", ""
        ),
        "gcg_suffix_sha256": hashlib.sha256(
            str(getattr(ATTACKS.get("gcg"), "universal_suffix", "")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "pair_max_rounds": getattr(ATTACKS.get("pair"), "max_rounds", ""),
        "pair_attacker_model": getattr(
            ATTACKS.get("pair"), "attacker_model", ""
        ),
        "pair_attacker_temperature": getattr(
            ATTACKS.get("pair"), "attacker_temperature", ""
        ),
        "pair_attacker_max_tokens": getattr(
            ATTACKS.get("pair"), "attacker_max_tokens", ""
        ),
        "pair_seed": getattr(ATTACKS.get("pair"), "seed", ""),
        "pair_target_temperature": config.MODEL_TEMPERATURE,
        "pair_target_max_tokens": config.MODEL_MAX_TOKENS,
        "pair_stopping_oracle": "sample_safe_unsafe_refusal_first",
    }


def _validate_existing_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    cells: list[tuple[str, str, str, str]],
) -> tuple[str | None, int]:
    if not manifest_path.exists():
        return None, 0
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "model": args.model,
        "quick": args.quick,
        "full": args.full,
        "dry_run": args.dry_run,
        "judges": args.judges,
        "jbb_provider": args.jbb_provider,
    }
    for field, expected in checks.items():
        if existing.get(field) != expected:
            raise ValueError(
                f"Cannot resume {manifest_path.parent}: manifest {field}="
                f"{existing.get(field)!r}, requested {expected!r}"
            )
    existing_cells = [
        (
            str(cell.get("prompt_set", "")),
            str(cell.get("batch", "")),
            str(cell.get("attack", "")),
            str(cell.get("defenses", "")),
        )
        for cell in existing.get("cells", [])
    ]
    if existing_cells and not set(existing_cells).issubset(set(cells)):
        raise ValueError("Cannot resume: manifest cells do not match the requested matrix")
    schema_version = int(existing.get("schema_version", 1))
    if schema_version >= 2:
        nested_checks = {
            "target_settings": {
                "temperature": config.MODEL_TEMPERATURE,
                "seed": config.MODEL_SEED,
                "max_tokens": config.MODEL_MAX_TOKENS,
                "deepinception_max_tokens": config.DEEPINCEPTION_MODEL_MAX_TOKENS,
            },
            "defense_settings": _defense_settings(),
            "attack_settings": _attack_settings(),
        }
        for field, expected in nested_checks.items():
            if existing.get(field) != expected:
                raise ValueError(
                    f"Cannot resume: manifest {field} differs from current settings"
                )
        expected_batches = {
            batch: _batch_provenance(batch)
            for batch in sorted({cell[1] for cell in cells})
        }
        if existing.get("batch_provenance") != expected_batches:
            raise ValueError(
                "Cannot resume: prompt batch hashes differ from the manifest"
            )
    return str(existing.get("created_at", "")) or None, schema_version


def run_matrix(args: argparse.Namespace) -> Path:
    output_dir = (args.output_dir or _default_output_dir()).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cells = _matrix_cells(args)
    manifest_path = output_dir / "manifest.json"
    created_at, checkpoint_schema = _validate_existing_manifest(
        manifest_path, args, cells
    )
    harmful_cells = sum(prompt_set == "harmful" for prompt_set, *_ in cells)
    benign_cells = len(cells) - harmful_cells
    print(
        f"Running {len(cells)} matrix cells: model={args.model}, "
        f"harmful={harmful_cells}, benign={benign_cells}, "
        f"prompts/cell={10 if args.quick else 100 if args.full else 50}"
    )
    print(f"Final judges: {', '.join(args.judges)}")
    print(f"Results: {output_dir}\n")

    source_paths: list[Path] = []
    manifest_cells: list[dict[str, object]] = []
    checkpoint_paths: list[Path] = []
    for index, (prompt_set, batch, attack, defense_stack) in enumerate(cells, 1):
        preferred_path = runs_dir / _cell_name(
            index, prompt_set, attack, defense_stack
        )
        checkpoint_path = _matching_checkpoint(
            preferred_path,
            # A manifest predating schema 2 cannot prove decoding, prompt, or
            # defense settings. Reusing it would mix incomparable cells.
            allow_existing=checkpoint_schema >= 2,
        )
        checkpoint_paths.append(checkpoint_path)
        rows = _validated_checkpoint_rows(
            checkpoint_path,
            model=args.model,
            batch=batch,
            prompt_set=prompt_set,
            attack=attack,
            defenses=defense_stack,
        )
        expected_count = len(load_batch(batch))
        manifest_cells.append(
            {
                "index": index,
                "attack": attack,
                "defenses": defense_stack,
                "prompt_set": prompt_set,
                "batch": batch,
                "source_csv": str(checkpoint_path),
                "expected_responses": expected_count,
                "completed_responses": len(rows),
                "status": "complete" if len(rows) == expected_count else (
                    "partial" if rows else "pending"
                ),
            }
        )

    manifest: dict[str, object] = {
        "schema_version": 2,
        "created_at": created_at or datetime.now().astimezone().isoformat(),
        "model": args.model,
        "batches": sorted({cell[1] for cell in cells}),
        "batch_provenance": {
            batch: _batch_provenance(batch) for batch in sorted({cell[1] for cell in cells})
        },
        "quick": args.quick,
        "full": args.full,
        "dry_run": args.dry_run,
        "judges": args.judges,
        "jbb_provider": args.jbb_provider,
        "target_settings": {
            "temperature": config.MODEL_TEMPERATURE,
            "seed": config.MODEL_SEED,
            "max_tokens": config.MODEL_MAX_TOKENS,
            "deepinception_max_tokens": config.DEEPINCEPTION_MODEL_MAX_TOKENS,
        },
        "defense_settings": _defense_settings(),
        "attack_settings": _attack_settings(),
        "code": _git_provenance(),
        "cells": manifest_cells,
    }
    _write_manifest(manifest_path, manifest)

    for index, ((prompt_set, batch, attack, defense_stack), cached_path) in enumerate(
        zip(cells, checkpoint_paths), 1
    ):
        print(
            f"\n=== Matrix cell {index}/{len(cells)}: "
            f"set={prompt_set}, attack={attack}, defenses={defense_stack} ==="
        )
        rows = _validated_checkpoint_rows(
            cached_path,
            model=args.model,
            batch=batch,
            prompt_set=prompt_set,
            attack=attack,
            defenses=defense_stack,
        )
        expected_count = len(load_batch(batch))
        if len(rows) == expected_count:
            print(f"checkpoint complete; skipping generation: {cached_path}")
            source_paths.append(cached_path)
            continue

        command = [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            "--model",
            args.model,
            "--batch",
            batch,
            "--attack",
            attack,
            "--defense",
            defense_stack,
            # Generation is cached once; the requested heavyweight judges run
            # over all completed cells together below.
            "--judge",
            "sample_safe_unsafe",
            "--output-csv",
            str(cached_path),
        ]
        if cached_path.exists():
            command.append("--resume")
        if args.dry_run:
            command.append("--dry-run")

        try:
            _run_and_find_csv(command, expected_path=cached_path)
        except Exception:
            partial_rows = _validated_checkpoint_rows(
                cached_path,
                model=args.model,
                batch=batch,
                prompt_set=prompt_set,
                attack=attack,
                defenses=defense_stack,
            )
            manifest_cells[index - 1]["completed_responses"] = len(partial_rows)
            manifest_cells[index - 1]["status"] = "partial" if partial_rows else "failed"
            _write_manifest(manifest_path, manifest)
            raise

        completed_rows = _validated_checkpoint_rows(
            cached_path,
            model=args.model,
            batch=batch,
            prompt_set=prompt_set,
            attack=attack,
            defenses=defense_stack,
        )
        if len(completed_rows) != expected_count:
            raise RuntimeError(
                f"Cell {index} completed with {len(completed_rows)}/{expected_count} rows"
            )
        source_paths.append(cached_path)
        manifest_cells[index - 1]["completed_responses"] = len(completed_rows)
        manifest_cells[index - 1]["status"] = "complete"
        _write_manifest(manifest_path, manifest)

    if args.dry_run:
        print(f"\nDry run complete; cached CSVs and manifest: {output_dir}")
        return output_dir

    evaluate_matrix(
        source_paths,
        output_dir,
        prompt_set_override="auto",
        jbb_provider=args.jbb_provider,
        run_jbb=any(
            name in args.judges
            for name in ("jailbreak_bench_llama8b", "jbb_refusal_llama3_8b")
        ),
        run_harmbench="harmbench_mistral" in args.judges,
        resume=args.resume,
    )
    return output_dir


def main() -> None:
    configure_utf8_stdio()
    run_matrix(parse_args())


if __name__ == "__main__":
    main()
