import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core import config
from judges.jailbreak_bench_llama8b import (
    _REFUSAL_PROMPT,
    JailbreakBenchLlama8bJudge,
    JailbreakBenchRefusalJudge,
)
from judges.runtime import JudgeResult


class JailbreakBenchRefusalJudgeTests(unittest.TestCase):
    def test_prompt_matches_the_pinned_official_literal(self) -> None:
        digest = hashlib.sha256(_REFUSAL_PROMPT.encode("utf-8")).hexdigest()
        self.assertEqual(
            digest,
            "987fdaaebb56135b745b6b23a9e0148a9052b340b7d77f3c7af39bb97dc0835b",
        )

    def test_yes_means_refused_and_no_means_not_refused(self) -> None:
        judge = JailbreakBenchRefusalJudge(
            batch_classifier=lambda prompts, responses: ["Yes", "No."],
        )
        results = judge.evaluate_batch(["p1", "p2"], ["r1", "r2"])
        self.assertEqual(
            results,
            [
                JudgeResult(label="Refused", score=1.0),
                JudgeResult(label="Not Refused", score=0.0),
            ],
        )

    def test_original_request_and_response_are_both_passed(self) -> None:
        seen = []

        def classify(prompts, responses):
            seen.append((list(prompts), list(responses)))
            return [False]

        result = JailbreakBenchRefusalJudge(batch_classifier=classify).evaluate(
            "original request", "target response"
        )
        self.assertEqual(seen, [(["original request"], ["target response"])])
        self.assertEqual(result.label, "Not Refused")

    def test_invalid_decision_is_rejected(self) -> None:
        judge = JailbreakBenchRefusalJudge(
            batch_classifier=lambda prompts, responses: ["maybe"],
        )
        with self.assertRaisesRegex(RuntimeError, "expected Yes or No"):
            judge.evaluate("prompt", "response")

    def test_local_provider_uses_deterministic_short_ollama_call(self) -> None:
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content='{"refused": true}')
        with patch("langchain_ollama.ChatOllama", return_value=model) as constructor:
            result = JailbreakBenchRefusalJudge(provider="ollama").evaluate(
                "request text", "response text"
            )

        constructor.assert_called_once_with(
            model=config.JBB_REFUSAL_OLLAMA_MODEL,
            temperature=0,
            seed=config.MODEL_SEED,
            num_predict=config.JBB_REFUSAL_OLLAMA_MAX_TOKENS,
            format={
                "type": "object",
                "properties": {"refused": {"type": "boolean"}},
                "required": ["refused"],
                "additionalProperties": False,
            },
        )
        messages = model.invoke.call_args.args[0]
        classifier_prompt = messages[1][1]
        self.assertEqual(messages[0][0], "system")
        self.assertIn("request text", classifier_prompt)
        self.assertIn("response text", classifier_prompt)
        self.assertEqual(result, JudgeResult(label="Refused", score=1.0))

    def test_together_provider_uses_official_model_route(self) -> None:
        api_response = Mock()
        api_response.raise_for_status.return_value = None
        api_response.json.return_value = {
            "choices": [{"message": {"content": "No"}}]
        }
        with (
            patch.object(config, "TOGETHER_API_KEY", "test-key"),
            patch("requests.post", return_value=api_response) as post,
        ):
            result = JailbreakBenchRefusalJudge(provider="together").evaluate(
                "request", "response"
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], config.JBB_REFUSAL_TOGETHER_MODEL)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(result, JudgeResult(label="Not Refused", score=0.0))

    def test_old_class_name_remains_compatible(self) -> None:
        self.assertIs(JailbreakBenchLlama8bJudge, JailbreakBenchRefusalJudge)


if __name__ == "__main__":
    unittest.main()
