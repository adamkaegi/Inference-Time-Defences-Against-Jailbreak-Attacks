"""Evaluate cached harness CSVs and build row-level plus aggregate reports."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core import config  # noqa: E402
from core.evaluation_reporting import (  # noqa: E402
    MATRIX_FIELDNAMES,
    format_matrix_digest,
    summarize_evaluated_rows,
    write_csv_rows,
)
from judges.harmbench_mistral import HarmBenchMistralJudge  # noqa: E402
from judges.jailbreak_bench_llama8b import (  # noqa: E402
    JailbreakBenchRefusalJudge,
)
from judges.runtime import evaluate_judge_batch  # noqa: E402
from prompts import available_batches, load_batch  # noqa: E402

EVALUATED_FIELDNAMES = [
    "source_csv",
    "run_id",
    "batch",
    "prompt_set",
    "prompt_index",
    "expected_responses",
    "model",
    "attack",
    "defenses",
    "input",
    "attacked_input",
    "output",
    "attack_latency_seconds",
    "response_pipeline_latency_seconds",
    "latency_seconds",
    "original_judge",
    "original_judge_label",
    "original_judge_score",
    "harmbench_status",
    "harmbench_label",
    "harmbench_score",
    "harmbench_behavior_success",
    "harmbench_batch_latency_seconds",
    "harmbench_error",
    "harmbench_model",
    "harmbench_revision",
    "jbb_refusal_status",
    "jbb_refusal_label",
    "jbb_refusal_score",
    "jbb_refused",
    "jbb_no_refusal",
    "jbb_refusal_batch_latency_seconds",
    "jbb_refusal_error",
    "jbb_refusal_provider",
    "jbb_refusal_model",
]

CHECKPOINT_IDENTITY_FIELDS = [
    "source_csv",
    "run_id",
    "batch",
    "prompt_set",
    "prompt_index",
    "expected_responses",
    "model",
    "attack",
    "defenses",
    "input",
    "attacked_input",
    "output",
    "attack_latency_seconds",
    "response_pipeline_latency_seconds",
    "latency_seconds",
    "original_judge",
    "original_judge_label",
    "original_judge_score",
]
HARMBENCH_FIELDS = [name for name in EVALUATED_FIELDNAMES if name.startswith("harmbench_")]
JBB_FIELDS = [name for name in EVALUATED_FIELDNAMES if name.startswith("jbb_refusal_")]


def _infer_batch(source_path: Path) -> str:
    stem = source_path.stem
    for batch in sorted(available_batches(), key=len, reverse=True):
        if stem.startswith(f"run-{batch}-"):
            return batch
    return "single" if stem.startswith("run-single-") else "unknown"


def _infer_prompt_set(batch: str, source_path: Path) -> str:
    searchable = f"{batch} {source_path.stem}".lower()
    if "jailbreakbench_harmful" in searchable:
        return "harmful"
    if "jailbreakbench_benign" in searchable:
        return "benign"
    return "unknown"


def _expected_count(batch: str, fallback: int) -> int:
    if batch in available_batches():
        return len(load_batch(batch))
    return fallback


def _normalize_source_rows(
    source_path: Path,
    prompt_set_override: str,
    force: bool,
) -> list[dict[str, Any]]:
    with source_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        source_rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    missing = {"input", "output"}.difference(fieldnames)
    if missing:
        raise ValueError(
            f"{source_path} is missing required column(s): {', '.join(sorted(missing))}"
        )

    fallback_batch = _infer_batch(source_path)
    normalized: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows, 1):
        batch = str(source.get("batch", "")).strip() or fallback_batch
        prompt_set = (
            prompt_set_override
            if prompt_set_override != "auto"
            else str(source.get("prompt_set", "")).strip()
            or _infer_prompt_set(batch, source_path)
        )
        expected = _expected_count(batch, len(source_rows))
        original_judge = str(source.get("judge", "")).strip()
        original_label = str(source.get("judge_label", "")).strip()
        original_score = str(source.get("judge_score", "")).strip()

        row: dict[str, Any] = {
            "source_csv": source_path.name,
            "run_id": source_path.stem,
            "batch": batch,
            "prompt_set": prompt_set,
            "prompt_index": source.get("prompt_index", "") or index,
            "expected_responses": expected,
            "model": source.get("model", ""),
            "attack": source.get("attack", ""),
            "defenses": source.get("defenses", ""),
            "input": source.get("input", ""),
            "attacked_input": source.get("attacked_input", ""),
            "output": source.get("output", ""),
            "attack_latency_seconds": source.get("attack_latency_seconds", ""),
            "response_pipeline_latency_seconds": source.get(
                "response_pipeline_latency_seconds", ""
            ),
            "latency_seconds": source.get("latency_seconds", ""),
            "original_judge": original_judge,
            "original_judge_label": original_label,
            "original_judge_score": original_score,
            "harmbench_status": "",
            "harmbench_label": "",
            "harmbench_score": "",
            "harmbench_behavior_success": "",
            "harmbench_batch_latency_seconds": "",
            "harmbench_error": "",
            "harmbench_model": config.HARMBENCH_MODEL,
            "harmbench_revision": config.HARMBENCH_MODEL_REVISION,
            "jbb_refusal_status": "",
            "jbb_refusal_label": "",
            "jbb_refusal_score": "",
            "jbb_refused": "",
            "jbb_no_refusal": "",
            "jbb_refusal_batch_latency_seconds": "",
            "jbb_refusal_error": "",
            "jbb_refusal_provider": "",
            "jbb_refusal_model": "",
        }

        if prompt_set == "benign":
            row["harmbench_status"] = "not_applicable"
        elif prompt_set == "unknown":
            row["harmbench_status"] = "unknown_prompt_set"

        if not force and original_judge == "harmbench_mistral_7b_val_cls":
            if original_label.lower() in {"safe", "unsafe"}:
                unsafe = original_label.lower() == "unsafe"
                row.update(
                    {
                        "harmbench_status": "reused",
                        "harmbench_label": "Unsafe" if unsafe else "Safe",
                        "harmbench_score": original_score or ("1.000000" if unsafe else "0.000000"),
                        "harmbench_behavior_success": "1" if unsafe else "0",
                        "harmbench_batch_latency_seconds": source.get(
                            "judge_batch_latency_seconds", ""
                        ),
                    }
                )

        if not force and original_judge == "jbb_refusal_llama3_8b":
            if original_label.lower() in {"refused", "not refused"}:
                refused = original_label.lower() == "refused"
                row.update(
                    {
                        "jbb_refusal_status": "reused",
                        "jbb_refusal_label": "Refused" if refused else "Not Refused",
                        "jbb_refusal_score": original_score or ("1.000000" if refused else "0.000000"),
                        "jbb_refused": "1" if refused else "0",
                        "jbb_no_refusal": "0" if refused else "1",
                        "jbb_refusal_batch_latency_seconds": source.get(
                            "judge_batch_latency_seconds", ""
                        ),
                        "jbb_refusal_provider": source.get("judge_provider", "unknown"),
                        "jbb_refusal_model": source.get("judge_model", "unknown"),
                    }
                )
        normalized.append(row)
    return normalized


def collect_source_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            candidates = sorted(input_path.glob("run-*.csv"))
            paths.extend(
                path
                for path in candidates
                if not any(
                    marker in path.stem
                    for marker in ("evaluated_rows", "experiment_matrix", "summary")
                )
            )
        elif input_path.is_file():
            paths.append(input_path)
        else:
            raise FileNotFoundError(input_path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if not unique:
        raise ValueError("No run CSV files were found")
    return unique


def _stop_ollama_model(model_name: str) -> None:
    if not model_name:
        return
    try:
        result = subprocess.run(
            ["ollama", "stop", model_name],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("warning: Ollama is unavailable; could not release model VRAM")
        return
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        print(f"warning: could not unload Ollama model '{model_name}': {detail}")


def _release_ollama_models(rows: list[dict[str, Any]]) -> None:
    """Release target, defense, judge, and PAIR models before a large judge."""
    models = {
        str(row.get("model", "")).strip() for row in rows
    }
    models.update(config.SUPPORT_MODELS)
    models.add(config.PAIR_ATTACKER_MODEL)
    for model_name in sorted(models - {""}):
        _stop_ollama_model(model_name)


def _expected_jbb_provenance(
    provider: str,
    judge: JailbreakBenchRefusalJudge | None,
) -> tuple[str, str]:
    if judge is not None:
        return str(judge.provider), str(judge.model_name)
    normalized = provider.strip().lower()
    model = (
        config.JBB_REFUSAL_TOGETHER_MODEL
        if normalized == "together"
        else config.JBB_REFUSAL_OLLAMA_MODEL
    )
    return normalized, model


def _load_evaluation_checkpoint(
    checkpoint_path: Path,
    fresh_rows: list[dict[str, Any]],
    *,
    run_jbb: bool,
    run_harmbench: bool,
    force: bool,
    jbb_provider: str,
    jbb_model: str,
) -> list[dict[str, Any]]:
    """Validate a row-level report and reuse only compatible judge results."""
    with checkpoint_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(EVALUATED_FIELDNAMES).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Evaluation checkpoint {checkpoint_path} is missing column(s): "
                f"{', '.join(sorted(missing))}"
            )
        cached_rows = [dict(row) for row in reader]

    if len(cached_rows) != len(fresh_rows):
        raise ValueError(
            f"Evaluation checkpoint {checkpoint_path} has {len(cached_rows)} rows; "
            f"the selected source CSVs have {len(fresh_rows)}"
        )

    merged_rows: list[dict[str, Any]] = []
    for index, (fresh, cached) in enumerate(zip(fresh_rows, cached_rows), 1):
        for field in CHECKPOINT_IDENTITY_FIELDS:
            if str(cached.get(field, "")) != str(fresh.get(field, "")):
                raise ValueError(
                    f"Evaluation checkpoint {checkpoint_path} row {index} no longer "
                    f"matches source field {field!r}"
                )

        harmbench_label = str(cached.get("harmbench_label", "")).strip()
        if harmbench_label and harmbench_label.lower() not in {"safe", "unsafe"}:
            raise ValueError(
                f"Evaluation checkpoint row {index} has invalid HarmBench label "
                f"{harmbench_label!r}"
            )
        if harmbench_label and run_harmbench and not force:
            provenance = (
                str(cached.get("harmbench_model", "")),
                str(cached.get("harmbench_revision", "")),
            )
            expected = (config.HARMBENCH_MODEL, config.HARMBENCH_MODEL_REVISION)
            if provenance != expected:
                raise ValueError(
                    "Evaluation checkpoint HarmBench provenance differs from the "
                    "current pinned evaluator; use --force to re-evaluate it"
                )

        refusal_label = str(cached.get("jbb_refusal_label", "")).strip()
        if refusal_label and refusal_label.lower() not in {"refused", "not refused"}:
            raise ValueError(
                f"Evaluation checkpoint row {index} has invalid JBB refusal label "
                f"{refusal_label!r}"
            )
        if refusal_label and run_jbb and not force:
            provenance = (
                str(cached.get("jbb_refusal_provider", "")),
                str(cached.get("jbb_refusal_model", "")),
            )
            if provenance != (jbb_provider, jbb_model):
                raise ValueError(
                    "Evaluation checkpoint JBB provider/model differs from the "
                    "requested evaluator; use --force to re-evaluate it"
                )

        merged = dict(fresh)
        if not (force and run_harmbench):
            merged.update({field: cached.get(field, "") for field in HARMBENCH_FIELDS})
        if not (force and run_jbb):
            merged.update({field: cached.get(field, "") for field in JBB_FIELDS})
        merged_rows.append(merged)
    return merged_rows


def _evaluate_jbb(
    rows: list[dict[str, Any]],
    judge: JailbreakBenchRefusalJudge,
) -> int:
    missing = [row for row in rows if not row["jbb_refusal_label"]]
    if not missing:
        return 0
    start = time.perf_counter()
    try:
        results = evaluate_judge_batch(
            judge,
            [str(row["input"]) for row in missing],
            [str(row["output"]) for row in missing],
        )
    except Exception as exc:
        for row in missing:
            row["jbb_refusal_status"] = "error"
            row["jbb_refusal_error"] = str(exc)
        raise
    latency = time.perf_counter() - start
    for row, result in zip(missing, results):
        refused = result.label == "Refused"
        row.update(
            {
                "jbb_refusal_status": "evaluated",
                "jbb_refusal_label": result.label or "",
                "jbb_refusal_score": "" if result.score is None else f"{result.score:.6f}",
                "jbb_refused": "1" if refused else "0",
                "jbb_no_refusal": "0" if refused else "1",
                "jbb_refusal_batch_latency_seconds": f"{latency:.3f}",
                "jbb_refusal_provider": judge.provider,
                "jbb_refusal_model": judge.model_name,
                "jbb_refusal_error": "",
            }
        )
    return len(missing)


def _evaluate_harmbench(
    rows: list[dict[str, Any]],
    judge: HarmBenchMistralJudge,
) -> int:
    missing = [
        row
        for row in rows
        if row["prompt_set"] == "harmful" and not row["harmbench_label"]
    ]
    if not missing:
        return 0
    start = time.perf_counter()
    try:
        results = evaluate_judge_batch(
            judge,
            [str(row["input"]) for row in missing],
            [str(row["output"]) for row in missing],
        )
    except Exception as exc:
        for row in missing:
            row["harmbench_status"] = "error"
            row["harmbench_error"] = str(exc)
        raise
    latency = time.perf_counter() - start
    for row, result in zip(missing, results):
        unsafe = result.label == "Unsafe"
        row.update(
            {
                "harmbench_status": "evaluated",
                "harmbench_label": result.label or "",
                "harmbench_score": "" if result.score is None else f"{result.score:.6f}",
                "harmbench_behavior_success": "1" if unsafe else "0",
                "harmbench_batch_latency_seconds": f"{latency:.3f}",
                "harmbench_error": "",
            }
        )
    return len(missing)


def save_reports(output_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = summarize_evaluated_rows(rows)
    digest = format_matrix_digest(summaries)
    write_csv_rows(output_dir / "evaluated_rows.csv", rows, EVALUATED_FIELDNAMES)
    write_csv_rows(output_dir / "experiment_matrix.csv", summaries, MATRIX_FIELDNAMES)
    summary_path = output_dir / "summary.md"
    temporary_path = summary_path.with_name(
        f".{summary_path.name}.{uuid4().hex[:8]}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as summary_file:
            summary_file.write("# Evaluation digest\n\n" + "\n".join(digest) + "\n")
            summary_file.flush()
            os.fsync(summary_file.fileno())
        os.replace(temporary_path, summary_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return summaries


def evaluate_matrix(
    source_paths: list[Path],
    output_dir: Path,
    prompt_set_override: str = "auto",
    jbb_provider: str = config.JBB_REFUSAL_PROVIDER,
    run_jbb: bool = True,
    run_harmbench: bool = True,
    force: bool = False,
    resume: bool = False,
    jbb_judge: JailbreakBenchRefusalJudge | None = None,
    harmbench_judge: HarmBenchMistralJudge | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_paths = [
        output_dir / "evaluated_rows.csv",
        output_dir / "experiment_matrix.csv",
        output_dir / "summary.md",
    ]
    existing_targets = [path for path in target_paths if path.exists()]
    if existing_targets and not resume:
        raise FileExistsError(
            f"Output directory already contains an evaluation report: {output_dir}"
        )

    fresh_rows: list[dict[str, Any]] = []
    for source_path in source_paths:
        fresh_rows.extend(_normalize_source_rows(source_path, prompt_set_override, force))

    checkpoint_path = output_dir / "evaluated_rows.csv"
    jbb_expected_provider, jbb_expected_model = _expected_jbb_provenance(
        jbb_provider,
        jbb_judge,
    )
    if resume and existing_targets:
        if not checkpoint_path.exists():
            raise ValueError(
                "Cannot resume evaluation: derived reports exist without the "
                "authoritative evaluated_rows.csv checkpoint"
            )
        rows = _load_evaluation_checkpoint(
            checkpoint_path,
            fresh_rows,
            run_jbb=run_jbb,
            run_harmbench=run_harmbench,
            force=force,
            jbb_provider=jbb_expected_provider,
            jbb_model=jbb_expected_model,
        )
    else:
        rows = fresh_rows

    try:
        if run_jbb:
            judge = jbb_judge or JailbreakBenchRefusalJudge(provider=jbb_provider)
            missing_jbb = any(not row["jbb_refusal_label"] for row in rows)
            if missing_jbb and judge.requires_target_model_unload:
                print("releasing Ollama models before JBB refusal judge...")
                _release_ollama_models(rows)
            evaluated = _evaluate_jbb(rows, judge)
            if evaluated and judge.provider == "ollama":
                _stop_ollama_model(judge.model_name)
            save_reports(output_dir, rows)

        if run_harmbench:
            judge = harmbench_judge or HarmBenchMistralJudge()
            missing_harmbench = any(
                row["prompt_set"] == "harmful" and not row["harmbench_label"]
                for row in rows
            )
            if missing_harmbench:
                print("releasing Ollama models before HarmBench...")
                _release_ollama_models(rows)
            _evaluate_harmbench(rows, judge)
    except Exception as exc:
        # Preserve every completed label even if the next evaluator fails.
        save_reports(output_dir, rows)
        raise RuntimeError(
            f"Evaluation failed; partial reports were saved in {output_dir}"
        ) from exc

    summaries = save_reports(output_dir, rows)
    print("\nEvaluation digest")
    for line in format_matrix_digest(summaries):
        print(line)
    print(f"\nrow-level CSV: {output_dir / 'evaluated_rows.csv'}")
    print(f"matrix CSV:    {output_dir / 'experiment_matrix.csv'}")
    print(f"digest:        {output_dir / 'summary.md'}")
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Run CSV file(s), or directories containing run-*.csv files.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--prompt-set",
        choices=("auto", "harmful", "benign"),
        default="auto",
        help="Override prompt-set inference when all inputs share one type.",
    )
    parser.add_argument(
        "--jbb-provider",
        choices=("ollama", "together"),
        default=config.JBB_REFUSAL_PROVIDER,
    )
    parser.add_argument("--skip-jbb", action="store_true")
    parser.add_argument("--skip-harmbench", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate labels already present in source CSVs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and continue a partial evaluated_rows.csv checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = collect_source_paths(args.inputs)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = PROJECT_ROOT / "outputs" / f"evaluation-{timestamp}-{uuid4().hex[:8]}"
    evaluate_matrix(
        source_paths,
        args.output_dir,
        prompt_set_override=args.prompt_set,
        jbb_provider=args.jbb_provider,
        run_jbb=not args.skip_jbb,
        run_harmbench=not args.skip_harmbench,
        force=args.force,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
