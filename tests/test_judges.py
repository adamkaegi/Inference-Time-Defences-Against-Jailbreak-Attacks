import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core import config
from judges.base import Judge
from judges.harmbench_mistral import (
    _MODEL_CACHE as HARMBENCH_MODEL_CACHE,
    HarmBenchMistralJudge,
)
from judges.runtime import JudgeResult, evaluate_judge, evaluate_judge_batch
from judges.sample_safe_unsafe import SampleSafeUnsafeJudge
from judges.strongreject import _JUDGE_PROMPT, _MODEL_CACHE, StrongRejectJudge


class _ResponseOnlyJudge(Judge):
    name = "response_only"

    def apply(self, text: str) -> str:
        return "saw response" if text == "response" else "wrong input"


class _PairwiseOnlyJudge:
    name = "pairwise_only"

    def evaluate(self, prompt: str, response: str) -> JudgeResult:
        return JudgeResult(label=f"{prompt}|{response}")


class JudgeContractTests(unittest.TestCase):
    def test_response_only_judges_remain_compatible(self) -> None:
        result = evaluate_judge(_ResponseOnlyJudge(), "original prompt", "response")

        self.assertEqual(result, JudgeResult(label="saw response"))

    def test_sample_judge_exposes_label_and_numeric_score(self) -> None:
        safe = evaluate_judge(SampleSafeUnsafeJudge(), "prompt", "I cannot assist.")
        unsafe = evaluate_judge(
            SampleSafeUnsafeJudge(), "prompt", "Here is a payload."
        )

        self.assertEqual(safe, JudgeResult(label="Safe", score=0.0))
        self.assertEqual(unsafe, JudgeResult(label="Unsafe", score=1.0))

    def test_sample_judge_does_not_mark_refusal_as_unsafe_keyword_hit(self) -> None:
        result = evaluate_judge(
            SampleSafeUnsafeJudge(),
            "prompt",
            "I cannot help with instructions that could cause harm.",
        )

        self.assertEqual(result, JudgeResult(label="Safe", score=0.0))

    def test_sample_judge_handles_common_apology_refusal_before_keywords(self) -> None:
        result = evaluate_judge(
            SampleSafeUnsafeJudge(),
            "prompt",
            "I'm sorry, but I am unable to provide instructions for harming someone.",
        )

        self.assertEqual(result, JudgeResult(label="Safe", score=0.0))

    def test_batch_lengths_must_match(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_judge_batch(_ResponseOnlyJudge(), ["one"], [])

    def test_pairwise_judge_does_not_need_dummy_apply(self) -> None:
        result = evaluate_judge_batch(
            _PairwiseOnlyJudge(), ["prompt"], ["response"]
        )

        self.assertEqual(result, [JudgeResult(label="prompt|response")])


class StrongRejectJudgeTests(unittest.TestCase):
    def test_v1_prompt_matches_pinned_upstream_literal(self) -> None:
        digest = hashlib.sha256(_JUDGE_PROMPT.encode("utf-8")).hexdigest()

        self.assertEqual(
            digest,
            "d90102b02d66b34a3dbf7d82789ce92310965503a481dc725232626bd7700937",
        )

    def test_pairwise_batch_scores_are_preserved(self) -> None:
        seen: list[tuple[list[str], list[str]]] = []

        def fake_evaluator(prompts, responses):
            seen.append((list(prompts), list(responses)))
            return [0.125, 0.875]

        judge = StrongRejectJudge(batch_evaluator=fake_evaluator)
        results = judge.evaluate_batch(["p1", "p2"], ["r1", "r2"])

        self.assertEqual(seen, [(["p1", "p2"], ["r1", "r2"])])
        self.assertEqual(results, [JudgeResult(score=0.125), JudgeResult(score=0.875)])

    def test_single_evaluation_passes_original_prompt(self) -> None:
        def fake_evaluator(prompts, responses):
            self.assertEqual(list(prompts), ["original behavior"])
            self.assertEqual(list(responses), ["target response"])
            return [0.25]

        result = StrongRejectJudge(batch_evaluator=fake_evaluator).evaluate(
            "original behavior", "target response"
        )

        self.assertEqual(result, JudgeResult(score=0.25))

    def test_invalid_scores_are_rejected(self) -> None:
        judge = StrongRejectJudge(batch_evaluator=lambda prompts, responses: [1.1])

        with self.assertRaises(RuntimeError):
            judge.evaluate("prompt", "response")

    def test_local_cpu_inference_requires_explicit_opt_in(self) -> None:
        judge = StrongRejectJudge(allow_cpu=False)

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "STRONGREJECT_ALLOW_CPU=1"):
                judge.evaluate("prompt", "response")

    def test_local_loader_pins_base_and_adapter_separately(self) -> None:
        import torch

        judge = StrongRejectJudge()
        base_model = Mock()
        adapted_model = Mock()
        tokenizer = Mock()

        with (
            patch.dict(_MODEL_CACHE, {}, clear=True),
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=base_model,
            ) as load_base,
            patch("peft.PeftModel.from_pretrained", return_value=adapted_model) as load_adapter,
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as load_tokenizer,
            patch.object(judge, "_score_batch", return_value=[0.5]),
        ):
            result = judge.evaluate("prompt", "response")

        self.assertEqual(result, JudgeResult(score=0.5))
        load_base.assert_called_once_with(
            config.STRONGREJECT_BASE_MODEL,
            revision=config.STRONGREJECT_BASE_REVISION,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        load_adapter.assert_called_once_with(
            base_model,
            config.STRONGREJECT_MODEL,
            revision=config.STRONGREJECT_MODEL_REVISION,
        )
        load_tokenizer.assert_called_once_with(
            config.STRONGREJECT_MODEL,
            revision=config.STRONGREJECT_MODEL_REVISION,
            padding_side="left",
            truncation_side="left",
        )

    def test_combined_prompt_is_capped_at_gemma_context_window(self) -> None:
        import torch

        class RecordingTokenizer:
            vocab = {value: index for index, value in enumerate(("1", "2", "3", "4", "5"))}

            def __init__(self) -> None:
                self.calls: list[tuple[object, dict[str, object]]] = []

            def __call__(self, value, **kwargs):
                self.calls.append((value, kwargs))
                if isinstance(value, str):
                    return {"input_ids": [0, 1]}
                return {
                    "input_ids": torch.tensor([[0, 1]]),
                    "attention_mask": torch.tensor([[1, 1]]),
                }

            def decode(self, token_ids):
                return "clipped response"

        class FakeModel:
            device = torch.device("cpu")
            config = SimpleNamespace(max_position_embeddings=64)

            def forward(self, input_ids, attention_mask, use_cache=False):
                return SimpleNamespace(logits=torch.zeros((1, input_ids.shape[1], 5)))

            __call__ = forward

        tokenizer = RecordingTokenizer()
        scores = StrongRejectJudge()._score_batch(
            ["very long behavior " * 10_000],
            ["response"],
            FakeModel(),
            tokenizer,
            torch,
        )

        combined_call = tokenizer.calls[1]
        self.assertEqual(scores, [0.5])
        self.assertEqual(
            combined_call[1],
            {
                "padding": True,
                "truncation": True,
                "max_length": 64,
                "return_tensors": "pt",
            },
        )


class HarmBenchMistralJudgeTests(unittest.TestCase):
    def test_yes_and_no_map_to_binary_unsafe_and_safe_results(self) -> None:
        seen: list[tuple[list[str], list[str]]] = []

        def fake_classifier(prompts, responses):
            seen.append((list(prompts), list(responses)))
            return ["Yes", "No"]

        judge = HarmBenchMistralJudge(batch_classifier=fake_classifier)
        results = judge.evaluate_batch(["behavior 1", "behavior 2"], ["r1", "r2"])

        self.assertEqual(
            seen,
            [(["behavior 1", "behavior 2"], ["r1", "r2"])],
        )
        self.assertEqual(
            results,
            [
                JudgeResult(label="Unsafe", score=1.0),
                JudgeResult(label="Safe", score=0.0),
            ],
        )

    def test_invalid_classifier_label_is_rejected(self) -> None:
        judge = HarmBenchMistralJudge(
            batch_classifier=lambda prompts, responses: ["Maybe"]
        )

        with self.assertRaisesRegex(RuntimeError, "expected Yes or No"):
            judge.evaluate("behavior", "response")

    def test_local_cpu_inference_requires_explicit_opt_in(self) -> None:
        judge = HarmBenchMistralJudge(allow_cpu=False)

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "HARMBENCH_ALLOW_CPU=1"):
                judge.evaluate("behavior", "response")

    def test_local_loader_uses_the_pinned_full_checkpoint(self) -> None:
        import torch

        judge = HarmBenchMistralJudge()
        model = Mock()
        tokenizer = Mock()

        with (
            patch.dict(HARMBENCH_MODEL_CACHE, {}, clear=True),
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ) as load_model,
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as load_tokenizer,
            patch.object(judge, "_classify_batch", return_value=["No"]),
        ):
            result = judge.evaluate("behavior", "response")

        self.assertEqual(result, JudgeResult(label="Safe", score=0.0))
        load_model.assert_called_once_with(
            config.HARMBENCH_MODEL,
            revision=config.HARMBENCH_MODEL_REVISION,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        load_tokenizer.assert_called_once_with(
            config.HARMBENCH_MODEL,
            revision=config.HARMBENCH_MODEL_REVISION,
        )


if __name__ == "__main__":
    unittest.main()
