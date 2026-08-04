import csv
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from core import config
from core.evaluation_reporting import (
    format_judge_digest,
    format_matrix_digest,
    summarize_evaluated_rows,
)
from judges.runtime import JudgeResult
from judges.harmbench_mistral import HarmBenchMistralJudge
from judges.jailbreak_bench_llama8b import JailbreakBenchRefusalJudge
from scripts.evaluate_matrix import evaluate_matrix


def _row(
    index: int,
    prompt_set: str,
    harmbench_label: str = "",
    refusal_label: str = "",
) -> dict[str, str]:
    return {
        "source_csv": f"{prompt_set}.csv",
        "run_id": f"{prompt_set}-run",
        "batch": f"jailbreakbench_{prompt_set}_4",
        "prompt_set": prompt_set,
        "prompt_index": str(index),
        "model": "qwen",
        "attack": "none",
        "defenses": "none",
        "expected_responses": "4",
        "output": f"response {index}",
        "harmbench_label": harmbench_label,
        "harmbench_status": "evaluated" if harmbench_label else "not_applicable",
        "jbb_refusal_label": refusal_label,
        "jbb_refusal_status": "evaluated" if refusal_label else "",
    }


class EvaluationReportingTests(unittest.TestCase):
    def test_harmful_summary_keeps_behavior_and_refusal_metrics_separate(self) -> None:
        rows = [
            _row(1, "harmful", "Unsafe", "Not Refused"),
            _row(2, "harmful", "Safe", "Not Refused"),
            _row(3, "harmful", "Safe", "Refused"),
            _row(4, "harmful", "Unsafe", "Refused"),
        ]
        summary = summarize_evaluated_rows(rows)[0]

        self.assertEqual(summary["harmbench_behavior_asr"], 0.5)
        self.assertEqual(summary["jbb_no_refusal_rate"], 0.5)
        self.assertEqual(summary["behavior_failure_and_no_refusal_count"], 1)
        self.assertEqual(summary["nonrefusing_but_unsuccessful_rate"], 0.5)
        self.assertEqual(summary["joint_judged_count"], 4)
        digest = "\n".join(format_matrix_digest([summary]))
        self.assertIn("Behavior successful despite refusal language: 1/4", digest)

    def test_missing_labels_and_duplicate_indices_are_not_reported_complete(self) -> None:
        rows = [
            _row(1, "harmful", "Unsafe", "Not Refused"),
            _row(1, "harmful"),
            _row(3, "harmful", "Safe", "Refused"),
            _row(4, "harmful", "Safe", "Refused"),
        ]

        summary = summarize_evaluated_rows(rows)[0]

        self.assertEqual(summary["run_status"], "partial")
        self.assertEqual(summary["duplicate_prompt_index_count"], 1)
        self.assertEqual(summary["harmbench_unknown_count"], 1)
        self.assertEqual(summary["jbb_unknown_count"], 1)

    def test_partial_judging_is_not_ranked_as_the_best_defense(self) -> None:
        rows = []
        for defenses in ("perplexity", "llama_guard_output", "perplexity,llama_guard_output"):
            for index in range(1, 5):
                harmbench = ""
                refusal = ""
                if defenses != "perplexity" or index == 1:
                    harmbench = "Unsafe" if defenses == "llama_guard_output" else "Safe"
                    refusal = "Not Refused"
                row = _row(index, "harmful", harmbench, refusal)
                row["source_csv"] = f"{defenses}.csv"
                row["run_id"] = defenses
                row["defenses"] = defenses
                rows.append(row)

        summaries = summarize_evaluated_rows(rows)
        partial = next(row for row in summaries if row["defenses"] == "perplexity")
        stack = next(row for row in summaries if "," in row["defenses"])

        self.assertEqual(partial["run_status"], "partial")
        self.assertNotEqual(stack["best_single_defense"], "perplexity")

    def test_debug_keyword_digest_is_explicitly_not_final_asr(self) -> None:
        digest = "\n".join(
            format_judge_digest(
                "sample_safe_unsafe",
                [JudgeResult(label="Unsafe", score=1.0)],
            )
        )

        self.assertIn("DEBUG KEYWORDS ONLY - NOT FINAL ASR", digest)

    def test_latency_and_stack_comparison_are_presentation_ready(self) -> None:
        rows = []
        for defenses, unsafe_count, latency in (
            ("none", 3, 1.0),
            ("perplexity", 1, 2.0),
            ("llama_guard_output", 2, 1.5),
            ("perplexity,smoothllm,llama_guard_output", 0, 5.0),
        ):
            for index in range(1, 5):
                row = _row(
                    index,
                    "harmful",
                    "Unsafe" if index <= unsafe_count else "Safe",
                    "Not Refused",
                )
                row["source_csv"] = f"{defenses}.csv"
                row["run_id"] = defenses
                row["defenses"] = defenses
                row["response_pipeline_latency_seconds"] = str(latency)
                rows.append(row)

        summaries = summarize_evaluated_rows(rows)
        stack = next(row for row in summaries if "," in row["defenses"])

        self.assertEqual(stack["added_response_latency_vs_no_defense_seconds"], 4.0)
        self.assertEqual(stack["best_single_defense"], "perplexity")
        self.assertEqual(stack["stack_asr_minus_best_single_percentage_points"], -25.0)
        digest = "\n".join(format_matrix_digest(summaries))
        self.assertIn("+4.000s vs no defense", digest)
        self.assertIn("Stack vs best single defense (perplexity): -25.0 ASR pp", digest)

    def test_benign_summary_reports_false_refusal_rate(self) -> None:
        rows = [
            _row(1, "benign", refusal_label="Refused"),
            _row(2, "benign", refusal_label="Not Refused"),
        ]
        for row in rows:
            row["expected_responses"] = "2"
        summary = summarize_evaluated_rows(rows)[0]

        self.assertEqual(summary["benign_false_refusal_count"], 1)
        self.assertEqual(summary["benign_false_refusal_rate"], 0.5)
        digest = "\n".join(format_matrix_digest([summary]))
        self.assertIn("false-refusal / over-blocking rate: 50.0%", digest)

    def test_cached_harmbench_results_are_reused_in_complete_report(self) -> None:
        tests_dir = Path(__file__).parent
        source_path = tests_dir / f"run-jailbreakbench_harmful_2-{uuid4().hex}.csv"
        output_dir = tests_dir
        report_paths = [
            output_dir / "evaluated_rows.csv",
            output_dir / "experiment_matrix.csv",
            output_dir / "summary.md",
        ]
        try:
            with source_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=[
                        "model",
                        "attack",
                        "defenses",
                        "judge",
                        "input",
                        "output",
                        "judge_label",
                        "judge_score",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "model": "qwen",
                        "attack": "none",
                        "defenses": "none",
                        "judge": "harmbench_mistral_7b_val_cls",
                        "input": "p1",
                        "output": "r1",
                        "judge_label": "Unsafe",
                        "judge_score": "1.0",
                    }
                )
                writer.writerow(
                    {
                        "model": "qwen",
                        "attack": "none",
                        "defenses": "none",
                        "judge": "harmbench_mistral_7b_val_cls",
                        "input": "p2",
                        "output": "r2",
                        "judge_label": "Safe",
                        "judge_score": "0.0",
                    }
                )

            jbb = JailbreakBenchRefusalJudge(
                batch_classifier=lambda prompts, responses: ["No", "Yes"]
            )

            def unexpected_harmbench(prompts, responses):
                raise AssertionError("cached HarmBench labels should be reused")

            harmbench = HarmBenchMistralJudge(batch_classifier=unexpected_harmbench)
            with patch("scripts.evaluate_matrix._stop_ollama_model"):
                summaries = evaluate_matrix(
                    [source_path],
                    output_dir,
                    jbb_judge=jbb,
                    harmbench_judge=harmbench,
                )

            self.assertEqual(summaries[0]["harmbench_behavior_asr"], 0.5)
            self.assertEqual(summaries[0]["jbb_no_refusal_rate"], 0.5)
            self.assertTrue((output_dir / "evaluated_rows.csv").exists())
            self.assertTrue((output_dir / "experiment_matrix.csv").exists())
            self.assertTrue((output_dir / "summary.md").exists())
        finally:
            source_path.unlink(missing_ok=True)
            for report_path in report_paths:
                report_path.unlink(missing_ok=True)

    def test_partial_evaluation_resumes_without_repeating_completed_jbb(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        source_path = project_root / "outputs" / f"run-single-{uuid4().hex}.csv"
        output_dir = project_root / "outputs" / f"evaluation-test-{uuid4().hex}"
        jbb_calls = 0

        try:
            with source_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["model", "attack", "defenses", "input", "output"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "model": "qwen",
                            "attack": "none",
                            "defenses": "none",
                            "input": "p1",
                            "output": "r1",
                        },
                        {
                            "model": "qwen",
                            "attack": "none",
                            "defenses": "none",
                            "input": "p2",
                            "output": "r2",
                        },
                    ]
                )

            first_jbb = JailbreakBenchRefusalJudge(
                batch_classifier=lambda prompts, responses: ["No", "Yes"]
            )
            failing_harmbench = HarmBenchMistralJudge(
                batch_classifier=lambda prompts, responses: (_ for _ in ()).throw(
                    RuntimeError("judge unavailable")
                )
            )
            with (
                patch("scripts.evaluate_matrix._release_ollama_models"),
                self.assertRaisesRegex(RuntimeError, "partial reports were saved"),
            ):
                evaluate_matrix(
                    [source_path],
                    output_dir,
                    prompt_set_override="harmful",
                    jbb_judge=first_jbb,
                    harmbench_judge=failing_harmbench,
                )

            def unexpected_jbb(prompts, responses):
                nonlocal jbb_calls
                jbb_calls += 1
                raise AssertionError("completed JBB labels should be reused")

            second_jbb = JailbreakBenchRefusalJudge(batch_classifier=unexpected_jbb)
            working_harmbench = HarmBenchMistralJudge(
                batch_classifier=lambda prompts, responses: ["Yes", "No"]
            )
            with patch("scripts.evaluate_matrix._release_ollama_models"):
                summaries = evaluate_matrix(
                    [source_path],
                    output_dir,
                    prompt_set_override="harmful",
                    jbb_judge=second_jbb,
                    harmbench_judge=working_harmbench,
                    resume=True,
                )

            self.assertEqual(jbb_calls, 0)
            self.assertEqual(summaries[0]["run_status"], "complete")
            self.assertEqual(summaries[0]["harmbench_behavior_asr"], 0.5)
            self.assertEqual(summaries[0]["jbb_no_refusal_rate"], 0.5)

            with source_path.open(newline="", encoding="utf-8") as csv_file:
                changed_rows = list(csv.DictReader(csv_file))
            changed_rows[1]["output"] = "changed cached response"
            with source_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(changed_rows[0]))
                writer.writeheader()
                writer.writerows(changed_rows)

            with self.assertRaisesRegex(ValueError, "source field 'output'"):
                evaluate_matrix(
                    [source_path],
                    output_dir,
                    prompt_set_override="harmful",
                    jbb_judge=second_jbb,
                    harmbench_judge=working_harmbench,
                    resume=True,
                )
        finally:
            source_path.unlink(missing_ok=True)
            if output_dir.exists():
                self.assertEqual(output_dir.parent, project_root / "outputs")
                shutil.rmtree(output_dir)


if __name__ == "__main__":
    unittest.main()
