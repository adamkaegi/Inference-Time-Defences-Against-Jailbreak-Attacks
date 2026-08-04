"""Apply a registered judge to a previously cached harness CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from judges import JUDGES  # noqa: E402
from judges.base import Judge  # noqa: E402
from judges.runtime import evaluate_judge_batch, judge_requires_target_unload  # noqa: E402
from core.evaluation_reporting import format_judge_digest  # noqa: E402


def _default_output_path(input_path: Path, judge_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = uuid4().hex[:8]
    return input_path.with_name(
        f"{input_path.stem}-{judge_name}-{timestamp}-{run_id}{input_path.suffix}"
    )


def _unload_ollama_model(model_name: str) -> None:
    """Release a cached run's target model before a local GPU judge loads."""
    result = subprocess.run(
        ["ollama", "stop", model_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        print(f"warning: could not unload Ollama model '{model_name}': {detail}")


def judge_csv(input_path: Path, output_path: Path, judge: Judge) -> Path:
    # Avoid an expensive evaluator run when its destination is already taken.
    # The exclusive open below still protects against a concurrent race.
    if output_path.exists():
        raise FileExistsError(output_path)

    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    required = {"input", "output"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    if judge_requires_target_unload(judge):
        model_names = sorted({row.get("model", "").strip() for row in rows} - {""})
        for model_name in model_names:
            print(f"unloading Ollama target {model_name} before local judge...")
            _unload_ollama_model(model_name)

    start_time = time.perf_counter()
    results = evaluate_judge_batch(
        judge,
        [row["input"] for row in rows],
        [row["output"] for row in rows],
    )
    judge_latency_seconds = time.perf_counter() - start_time
    if len(results) != len(rows):
        raise RuntimeError(
            f"Judge '{judge.name}' returned {len(results)} result(s) for {len(rows)} row(s)"
        )

    result_columns = [
        "judge",
        "judge_provider",
        "judge_model",
        "judge_label",
        "judge_score",
        "judge_batch_latency_seconds",
    ]
    for column in result_columns:
        if column not in fieldnames:
            fieldnames.append(column)

    for row, result in zip(rows, results):
        row["judge"] = judge.name
        row["judge_provider"] = str(getattr(judge, "provider", ""))
        row["judge_model"] = str(getattr(judge, "model_name", ""))
        row["judge_label"] = result.label or ""
        row["judge_score"] = "" if result.score is None else f"{result.score:.6f}"
        row["judge_batch_latency_seconds"] = f"{judge_latency_seconds:.3f}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    for line in format_judge_digest(judge.name, results):
        print(line)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, help="Cached harness CSV to judge.")
    parser.add_argument("--judge", default="strongreject", choices=JUDGES.keys())
    parser.add_argument("--output", type=Path, help="New output CSV; never overwrites input.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or _default_output_path(args.csv_path, args.judge)
    saved_path = judge_csv(args.csv_path, output_path, JUDGES[args.judge])
    print(f"saved judged csv: {saved_path}")


if __name__ == "__main__":
    main()
