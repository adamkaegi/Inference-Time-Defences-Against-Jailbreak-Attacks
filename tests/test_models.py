import unittest

import config
from defenses import DEFENSES
from defenses.smoothllm import SmoothLLMDefense
from judges import JUDGES


class ModelConfigurationTests(unittest.TestCase):
    def test_qwen_7b_is_the_default_and_in_target_suite(self) -> None:
        self.assertEqual(config.MODEL, "qwen2.5:7b-instruct")
        self.assertEqual(config.TARGET_MODELS[0], config.MODEL)

    def test_all_proposal_models_are_exposed(self) -> None:
        self.assertEqual(
            set(config.TARGET_MODELS),
            {
                "qwen2.5:7b-instruct",
                "dolphin-mistral:7b",
                "qwen2.5:3b",
                "llama3.2:3b",
            },
        )
        self.assertIn("llama-guard3:1b", config.SUPPORT_MODELS)

    def test_smoothllm_follows_cli_target_model(self) -> None:
        configured = DEFENSES["smoothllm"].for_model("dolphin-mistral:7b")

        self.assertIsInstance(configured, SmoothLLMDefense)
        self.assertEqual(configured.model_name, "dolphin-mistral:7b")
        self.assertIsNot(configured, DEFENSES["smoothllm"])

    def test_smoothllm_dry_run_does_not_call_ollama(self) -> None:
        configured = DEFENSES["smoothllm"].for_run(config.MODEL, dry_run=True)

        self.assertTrue(configured.dry_run)
        self.assertIsInstance(configured.apply("A short prompt"), str)

    def test_requested_harmbench_judge_is_registered(self) -> None:
        self.assertIn("harmbench_mistral_7b_val_cls", JUDGES)

    def test_wildguard_refusal_judge_is_registered(self) -> None:
        self.assertIn("wildguard_refusal", JUDGES)


if __name__ == "__main__":
    unittest.main()
