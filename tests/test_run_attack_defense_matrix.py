import unittest
import json
import shutil
import subprocess
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
from uuid import uuid4

from core import config
import core.run_attack_defense_matrix as matrix
from attacks.pair import PAIRAttack


class MatrixRunnerArgumentTests(unittest.TestCase):
    def test_requested_defaults(self) -> None:
        args = matrix.parse_args([])

        self.assertEqual(args.model, config.MODEL)
        self.assertEqual(args.judges, list(matrix.DEFAULT_JUDGES))
        self.assertEqual(args.attacks, list(matrix.DEFAULT_ATTACKS))
        self.assertEqual(args.defense_stacks, list(matrix.DEFAULT_DEFENSE_STACKS))
        self.assertFalse(args.quick)
        self.assertFalse(args.full)
        self.assertFalse(args.skip_benign)
        cells = matrix._matrix_cells(args)
        self.assertEqual(len(cells), 30)
        self.assertEqual(sum(cell[0] == "harmful" for cell in cells), 24)
        self.assertEqual(sum(cell[0] == "benign" for cell in cells), 6)
        self.assertNotIn("llama_guard_input", args.defense_stacks)
        self.assertTrue(
            all(cell[1].endswith("_50") for cell in cells)
        )
        self.assertIn(
            "perplexity,smoothllm,llama_guard_output",
            args.defense_stacks,
        )

    def test_quick_mode_and_stacked_defenses(self) -> None:
        args = matrix.parse_args(
            [
                "--quick",
                "--model",
                "qwen2.5:3b",
                "--attack",
                "gcg",
                "--defense",
                "self_reminder,perplexity,llama_guard_output",
            ]
        )

        self.assertTrue(args.quick)
        self.assertFalse(args.full)
        self.assertEqual(args.model, "qwen2.5:3b")
        self.assertEqual(args.attacks, ["gcg"])
        self.assertEqual(
            args.defense_stacks,
            ["self_reminder,perplexity,llama_guard_output"],
        )
        self.assertEqual(
            matrix._matrix_cells(args),
            [
                (
                    "harmful",
                    matrix.QUICK_HARMFUL_BATCH,
                    "gcg",
                    "self_reminder,perplexity,llama_guard_output",
                ),
                (
                    "benign",
                    matrix.QUICK_BENIGN_BATCH,
                    "none",
                    "self_reminder,perplexity,llama_guard_output",
                ),
            ],
        )

    def test_full_mode_uses_all_one_hundred_prompts(self) -> None:
        args = matrix.parse_args(["--full", "--attack", "none", "--defense", "none"])

        cells = matrix._matrix_cells(args)

        self.assertFalse(args.quick)
        self.assertTrue(args.full)
        self.assertEqual(cells[0][1], matrix.FULL_HARMFUL_BATCH)
        self.assertEqual(cells[1][1], matrix.FULL_BENIGN_BATCH)

    def test_none_cannot_be_mixed_into_a_stack(self) -> None:
        with self.assertRaises(SystemExit):
            matrix.parse_args(["--defense", "none,self_reminder"])

    def test_pair_uses_the_selected_matrix_model(self) -> None:
        configured = PAIRAttack(max_rounds=1).for_run(
            "dolphin-mistral:7b", dry_run=True
        )

        self.assertEqual(configured.target_model, "dolphin-mistral:7b")
        self.assertEqual(
            configured.attacker_temperature,
            config.PAIR_ATTACKER_TEMPERATURE,
        )
        self.assertEqual(configured.seed, config.MODEL_SEED)
        self.assertEqual(configured.apply("test prompt"), "test prompt")

    def test_pair_decoding_settings_are_recorded_in_manifest_settings(self) -> None:
        settings = matrix._attack_settings()

        self.assertEqual(
            settings["pair_attacker_temperature"],
            config.PAIR_ATTACKER_TEMPERATURE,
        )
        self.assertEqual(settings["pair_seed"], config.MODEL_SEED)
        self.assertEqual(settings["pair_target_max_tokens"], config.MODEL_MAX_TOKENS)

    def test_deepinception_has_a_larger_target_generation_budget(self) -> None:
        self.assertEqual(config.model_max_tokens_for_attack("pair"), 512)
        self.assertEqual(config.model_max_tokens_for_attack("gcg"), 512)
        self.assertEqual(config.model_max_tokens_for_attack("deepinception"), 2048)
        self.assertEqual(config.model_max_tokens_for_attack("template"), 2048)

    def test_unmanifested_v2_checkpoint_is_not_reused(self) -> None:
        path = matrix.PROJECT_ROOT / "outputs" / f"unmanifested-{uuid4().hex}.csv"
        try:
            path.touch()

            with self.assertRaisesRegex(ValueError, "without a compatible schema-v2"):
                matrix._matching_checkpoint(path, allow_existing=False)
        finally:
            path.unlink(missing_ok=True)

    def test_resume_requires_an_explicit_output_directory(self) -> None:
        with self.assertRaises(SystemExit):
            matrix.parse_args(["--resume"])

    def test_child_process_is_forced_to_utf8(self) -> None:
        environment = matrix._child_environment()

        self.assertEqual(environment["PYTHONIOENCODING"], "utf-8:backslashreplace")
        self.assertEqual(environment["PYTHONUTF8"], "1")
        self.assertEqual(environment["PYTHONUNBUFFERED"], "1")

    def test_parent_decodes_and_relays_utf8_child_output(self) -> None:
        expected = (matrix.PROJECT_ROOT / "outputs" / "expected.csv").resolve()
        process = Mock()
        process.stdout = iter(["model said 🌟\n", f"saved csv: {expected}\n"])
        process.wait.return_value = 0

        with (
            patch.object(matrix.subprocess, "Popen", return_value=process) as popen,
            patch("sys.stdout", new_callable=StringIO),
        ):
            result = matrix._run_and_find_csv(["python", "main.py"], expected)

        self.assertEqual(result, expected)
        self.assertEqual(popen.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(popen.call_args.kwargs["errors"], "replace")

    def test_manifest_survives_a_partial_cell_failure(self) -> None:
        output_root = (matrix.PROJECT_ROOT / "outputs").resolve()
        output_dir = output_root / f"matrix-test-{uuid4().hex}"
        self.assertEqual(output_dir.parent, output_root)
        try:
            args = matrix.parse_args(
                [
                    "--quick",
                    "--attack",
                    "none",
                    "--defense",
                    "none",
                    "--dry-run",
                    "--output-dir",
                    str(output_dir),
                ]
            )

            def fail_after_one_row(command, expected_path=None):
                self.assertTrue((output_dir / "manifest.json").exists())
                prompt = matrix.load_batch(matrix.QUICK_HARMFUL_BATCH)[0]
                row = {
                    "judge_provider": "",
                    "judge_model": "",
                    "batch": matrix.QUICK_HARMFUL_BATCH,
                    "prompt_set": "harmful",
                    "prompt_index": "1",
                    "input": prompt,
                    "attacked_input": prompt,
                    "output": "cached response 🌟",
                    "judge_label": "",
                    "judge_score": "",
                    "attack_latency_seconds": "0.000",
                    "response_pipeline_latency_seconds": "0.001",
                    "latency_seconds": "0.001",
                    "judge_batch_latency_seconds": "",
                }
                import main as main_module

                main_module._write_run_csv(
                    [row],
                    "none",
                    ["none"],
                    "sample_safe_unsafe",
                    config.MODEL,
                    matrix.QUICK_HARMFUL_BATCH,
                    csv_path=expected_path,
                )
                raise subprocess.CalledProcessError(1, command)

            with patch.object(matrix, "_run_and_find_csv", side_effect=fail_after_one_row):
                with self.assertRaises(subprocess.CalledProcessError):
                    matrix.run_matrix(args)

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["cells"][0]["status"], "partial")
            self.assertEqual(manifest["cells"][0]["completed_responses"], 1)
        finally:
            if output_dir.exists():
                self.assertEqual(output_dir.resolve().parent, output_root)
                shutil.rmtree(output_dir)


if __name__ == "__main__":
    unittest.main()
