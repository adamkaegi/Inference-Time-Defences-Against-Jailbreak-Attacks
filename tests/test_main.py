import argparse
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from core import config
import main
from judges.runtime import JudgeResult


class _CountingAttack:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def apply(self, prompt: str) -> str:
        self.calls += 1
        return f"{prompt}|attack-{self.calls}"


class _RunAwareDefense:
    name = "run_aware"
    stage = "input"

    def __init__(self) -> None:
        self.configure_calls: list[tuple[str, bool]] = []

    def for_run(self, model_name: str, dry_run: bool = False):
        self.configure_calls.append((model_name, dry_run))
        return self


class _FakeChain:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def invoke(self, text: str, **kwargs) -> str:
        self.inputs.append(text)
        return f"response to {text}"


class _FakeJudge:
    name = "fake"
    requires_target_model_unload = False

    def __init__(self, results=None) -> None:
        self.results = (
            [JudgeResult(label="Safe", score=0.0)] if results is None else results
        )
        self.calls = 0

    def evaluate_batch(self, prompts, responses):
        self.calls += 1
        return self.results


class MainIntegrationTests(unittest.TestCase):
    def _args(self, *, dry_run: bool) -> argparse.Namespace:
        return argparse.Namespace(
            prompt="",
            model="chosen-model",
            attack="counting",
            defense=["run_aware"],
            judge="fake",
            batch="general",
            dry_run=dry_run,
            output_csv=None,
            resume=False,
        )

    def test_attack_is_applied_once_and_exact_output_reaches_chain(self) -> None:
        attack = _CountingAttack()
        defense = _RunAwareDefense()
        judge = _FakeJudge()
        chain = _FakeChain()
        write_csv = Mock(return_value=Path("cached.csv"))

        with (
            patch.object(main, "parse_args", return_value=self._args(dry_run=True)),
            patch.dict(main.ATTACKS, {"counting": attack}),
            patch.dict(main.DEFENSES, {"run_aware": defense}),
            patch.dict(main.JUDGES, {"fake": judge}),
            patch.object(main, "build_response_chain", return_value=chain) as build_chain,
            patch.object(main, "_load_langfuse_handler", return_value=(None, None)),
            patch.object(main, "_write_run_csv", write_csv),
            patch.object(main, "load_batch") as load_batch,
        ):
            main.main()

        self.assertEqual(attack.calls, 1)
        self.assertEqual(chain.inputs, ["|attack-1"])
        self.assertEqual(defense.configure_calls, [("chosen-model", True)])
        build_chain.assert_called_once_with(
            [defense],
            "chosen-model",
            dry_run=True,
            max_tokens=config.MODEL_MAX_TOKENS,
        )
        load_batch.assert_not_called()
        self.assertEqual(judge.calls, 0)
        checkpoint_rows = write_csv.call_args_list[1].args[0]
        self.assertEqual(checkpoint_rows[0]["attacked_input"], "|attack-1")
        self.assertEqual(write_csv.call_count, 3)

    def test_short_judge_result_list_fails_after_checkpoint(self) -> None:
        attack = _CountingAttack()
        defense = _RunAwareDefense()
        judge = _FakeJudge(results=[])
        chain = _FakeChain()

        with (
            patch.object(main, "parse_args", return_value=self._args(dry_run=False)),
            patch.dict(main.ATTACKS, {"counting": attack}),
            patch.dict(main.DEFENSES, {"run_aware": defense}),
            patch.dict(main.JUDGES, {"fake": judge}),
            patch.object(main, "build_response_chain", return_value=chain),
            patch.object(main, "_load_langfuse_handler", return_value=(None, None)),
            patch.object(main, "_write_run_csv", return_value=Path("cached.csv")),
        ):
            with self.assertRaisesRegex(RuntimeError, "responses remain cached"):
                main.main()

        self.assertEqual(attack.calls, 1)

    def test_gpu_judge_unloads_selected_target_after_checkpoint(self) -> None:
        attack = _CountingAttack()
        defense = _RunAwareDefense()
        judge = _FakeJudge()
        judge.requires_target_model_unload = True
        chain = _FakeChain()

        with (
            patch.object(main, "parse_args", return_value=self._args(dry_run=False)),
            patch.dict(main.ATTACKS, {"counting": attack}),
            patch.dict(main.DEFENSES, {"run_aware": defense}),
            patch.dict(main.JUDGES, {"fake": judge}),
            patch.object(main, "build_response_chain", return_value=chain),
            patch.object(main, "_load_langfuse_handler", return_value=(None, None)),
            patch.object(main, "_write_run_csv", return_value=Path("cached.csv")),
            patch.object(main, "_unload_ollama_model") as unload,
        ):
            main.main()

        unload.assert_called_once_with("chosen-model")


class RunCheckpointTests(unittest.TestCase):
    def test_unicode_checkpoint_round_trips_as_a_valid_resume_prefix(self) -> None:
        path = Path(main.__file__).resolve().parent / "outputs" / f"checkpoint-test-{uuid4().hex}.csv"
        try:
            row = {
                "judge_provider": "",
                "judge_model": "",
                "batch": "batch",
                "prompt_set": "harmful",
                "prompt_index": "1",
                "input": "prompt one",
                "attacked_input": "attacked",
                "output": "response 🌟",
                "judge_label": "",
                "judge_score": "",
                "attack_latency_seconds": "0.1",
                "response_pipeline_latency_seconds": "0.2",
                "latency_seconds": "0.3",
                "judge_batch_latency_seconds": "",
            }
            main._write_run_csv(
                [row],
                "attack",
                ["defense"],
                "judge",
                "model",
                "batch",
                csv_path=path,
            )

            loaded = main._load_resume_rows(
                path,
                ["prompt one", "prompt two"],
                "attack",
                ["defense"],
                "judge",
                "model",
                "batch",
                "harmful",
            )

            self.assertEqual(loaded[0]["output"], "response 🌟")
        finally:
            path.unlink(missing_ok=True)

    def test_resume_rejects_a_different_prompt_batch(self) -> None:
        path = Path(main.__file__).resolve().parent / "outputs" / f"checkpoint-test-{uuid4().hex}.csv"
        try:
            row = {
                field: "" for field in main.RUN_FIELDNAMES
            }
            row.update(
                {
                    "batch": "batch",
                    "prompt_set": "harmful",
                    "prompt_index": "1",
                    "input": "old prompt",
                }
            )
            main._write_run_csv(
                [row],
                "attack",
                ["defense"],
                "judge",
                "model",
                "batch",
                csv_path=path,
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                main._load_resume_rows(
                    path,
                    ["new prompt"],
                    "attack",
                    ["defense"],
                    "judge",
                    "model",
                    "batch",
                    "harmful",
                )
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
