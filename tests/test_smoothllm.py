import unittest
from unittest.mock import patch

from defenses.base import Defense
from defenses.smoothllm import SmoothLLMDefense
from core.pipeline import build_response_chain


class _SuffixOutputDefense(Defense):
    name = "suffix"
    stage = "output"

    def apply(self, text: str) -> str:
        return f"{text}|checked"


class SmoothLLMTests(unittest.TestCase):
    def test_returns_a_response_from_the_majority_class(self) -> None:
        responses = iter(["jailbroken-1", "jailbroken-2", "jailbroken-3", "refusal"])
        generated_prompts = []

        def generate(prompt: str) -> str:
            generated_prompts.append(prompt)
            return next(responses)

        defense = SmoothLLMDefense(
            num_samples=4,
            perturbation_rate=0,
            seed=3,
            generator=generate,
            jailbreak_classifier=lambda response: response.startswith("jailbroken"),
        )

        result = defense.apply("original prompt")

        self.assertTrue(result.startswith("jailbroken"))
        self.assertNotEqual(result, "original prompt")
        self.assertEqual(len(generated_prompts), 4)

    def test_tie_selects_the_non_jailbroken_class(self) -> None:
        responses = iter(["jailbroken-1", "refusal-1", "jailbroken-2", "refusal-2"])
        defense = SmoothLLMDefense(
            num_samples=4,
            perturbation_rate=0,
            generator=lambda prompt: next(responses),
            jailbreak_classifier=lambda response: response.startswith("jailbroken"),
        )

        self.assertTrue(defense.apply("prompt").startswith("refusal"))

    def test_pipeline_does_not_make_an_extra_outer_model_call(self) -> None:
        generated = []

        def generate(prompt: str) -> str:
            generated.append(prompt)
            return f"response-{len(generated)}"

        smooth = SmoothLLMDefense(
            num_samples=3,
            perturbation_rate=0,
            generator=generate,
            jailbreak_classifier=lambda response: True,
        )
        with patch("core.pipeline._make_model", side_effect=AssertionError("outer call")):
            chain = build_response_chain(
                [smooth, _SuffixOutputDefense()],
                "unused-outer-model",
                dry_run=True,
            )
            result = chain.invoke("prompt")

        self.assertEqual(len(generated), 3)
        self.assertTrue(result.startswith("response-"))
        self.assertTrue(result.endswith("|checked"))

    def test_generation_failure_is_surfaced(self) -> None:
        def fail(prompt: str) -> str:
            raise RuntimeError("failed")

        defense = SmoothLLMDefense(num_samples=1, generator=fail)

        with self.assertRaisesRegex(RuntimeError, "failed"):
            defense.apply("prompt")

    def test_common_apology_refusal_is_not_counted_as_jailbroken(self) -> None:
        response = "I'm sorry, but I can't provide instructions for harming someone."

        self.assertFalse(SmoothLLMDefense._looks_jailbroken(response))

    def test_run_configuration_applies_attack_specific_token_budget(self) -> None:
        configured = SmoothLLMDefense().for_run(
            "chosen-model",
            dry_run=True,
            max_tokens=2048,
        )

        self.assertEqual(configured.max_tokens, 2048)


if __name__ == "__main__":
    unittest.main()
