import hashlib
import unittest
from unittest.mock import Mock, patch

import config
from judges.runtime import JudgeResult
from judges.wildguard import (
    _CLASSIFIER_PROMPT,
    _MODEL_CACHE,
    WildGuardRefusalJudge,
)


class WildGuardRefusalJudgeTests(unittest.TestCase):
    def tearDown(self) -> None:
        _MODEL_CACHE.clear()

    def test_prompt_matches_pinned_official_model_card_literal(self) -> None:
        digest = hashlib.sha256(_CLASSIFIER_PROMPT.encode("utf-8")).hexdigest()
        self.assertEqual(
            digest,
            "fa7f1c313ef1463a2fe249ff50c537adfbd84ff2fe82b3ffa72ee21124afb0d6",
        )

    def test_yes_means_refused_and_no_means_not_refused(self) -> None:
        judge = WildGuardRefusalJudge(
            batch_classifier=lambda prompts, responses: [
                "Harmful request: yes\n"
                "Response refusal: yes\n"
                "Harmful response: no",
                "Harmful request: no\n"
                "Response refusal: no\n"
                "Harmful response: no",
            ]
        )

        self.assertEqual(
            judge.evaluate_batch(["p1", "p2"], ["r1", "r2"]),
            [
                JudgeResult(label="Refused", score=1.0),
                JudgeResult(label="Not Refused", score=0.0),
            ],
        )

    def test_original_request_and_complete_response_are_passed(self) -> None:
        seen = []

        def classify(prompts, responses):
            seen.append((list(prompts), list(responses)))
            return [
                "Harmful request: no\n"
                "Response refusal: no\n"
                "Harmful response: no"
            ]

        result = WildGuardRefusalJudge(batch_classifier=classify).evaluate(
            "original request",
            "warning followed by a useful answer",
        )

        self.assertEqual(
            seen,
            [(
                ["original request"],
                ["warning followed by a useful answer"],
            )],
        )
        self.assertEqual(result.label, "Not Refused")

    def test_malformed_or_partial_output_is_rejected(self) -> None:
        judge = WildGuardRefusalJudge(
            batch_classifier=lambda prompts, responses: [
                "Response refusal: maybe"
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "invalid classification"):
            judge.evaluate("prompt", "response")

    def test_local_provider_loads_the_pinned_checkpoint(self) -> None:
        import torch

        model = Mock()
        model.device = torch.device("cpu")
        model.config.max_position_embeddings = 32768
        model.generate.return_value = torch.tensor([[1, 2, 3, 4]])

        tokenized = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        tokenized_wrapper = Mock()
        tokenized_wrapper.__getitem__ = Mock(side_effect=tokenized.__getitem__)

        tokenizer = Mock()
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = 0
        tokenizer.encode.return_value = [10]
        tokenizer.decode.return_value = "response"
        tokenizer.return_value = tokenized_wrapper
        tokenizer.batch_decode.return_value = [
            "Harmful request: no\n"
            "Response refusal: no\n"
            "Harmful response: no"
        ]

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch(
                "transformers.AutoModelForCausalLM.from_pretrained",
                return_value=model,
            ) as load_model,
            patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=tokenizer,
            ) as load_tokenizer,
        ):
            result = WildGuardRefusalJudge().evaluate("request", "response")

        load_model.assert_called_once_with(
            config.WILDGUARD_MODEL,
            revision=config.WILDGUARD_MODEL_REVISION,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        load_tokenizer.assert_called_once_with(
            config.WILDGUARD_MODEL,
            revision=config.WILDGUARD_MODEL_REVISION,
            use_fast=False,
        )
        self.assertEqual(result, JudgeResult(label="Not Refused", score=0.0))

    def test_unload_model_clears_the_cached_checkpoint(self) -> None:
        key = (config.WILDGUARD_MODEL, config.WILDGUARD_MODEL_REVISION)
        _MODEL_CACHE[key] = (Mock(), Mock())
        judge = WildGuardRefusalJudge(batch_classifier=lambda p, r: [])

        with patch("torch.cuda.is_available", return_value=False):
            judge.unload_model()

        self.assertNotIn(key, _MODEL_CACHE)


if __name__ == "__main__":
    unittest.main()
