import sys
import types
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from core import config
from attacks.base import Attack
from core.pipeline import _make_model, build_chain, build_response_chain


class _CountingAttack(Attack):
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def apply(self, prompt: str) -> str:
        self.calls += 1
        return f"{prompt}|attack-{self.calls}"


class PipelineTests(unittest.TestCase):
    def test_public_chain_applies_attack_once(self) -> None:
        attack = _CountingAttack()

        output = build_chain(attack, [], config.MODEL, dry_run=True).invoke("prompt")

        self.assertEqual(attack.calls, 1)
        self.assertIn("prompt|attack-1", output)

    def test_post_attack_chain_uses_exact_supplied_text(self) -> None:
        output = build_response_chain([], config.MODEL, dry_run=True).invoke("already attacked")

        self.assertIn("already attacked", output)

    def test_real_model_receives_deterministic_decoding_options(self) -> None:
        captured: dict[str, object] = {}
        fake_module = types.ModuleType("langchain_ollama")

        def fake_chat_ollama(**kwargs):
            captured.update(kwargs)
            return RunnableLambda(lambda value: value)

        fake_module.ChatOllama = fake_chat_ollama
        with patch.dict(sys.modules, {"langchain_ollama": fake_module}):
            _make_model("chosen-model", dry_run=False)

        self.assertEqual(
            captured,
            {
                "model": "chosen-model",
                "temperature": config.MODEL_TEMPERATURE,
                "seed": config.MODEL_SEED,
                "num_predict": config.MODEL_MAX_TOKENS,
            },
        )

    def test_real_model_accepts_an_attack_specific_token_budget(self) -> None:
        captured: dict[str, object] = {}
        fake_module = types.ModuleType("langchain_ollama")

        def fake_chat_ollama(**kwargs):
            captured.update(kwargs)
            return RunnableLambda(lambda value: value)

        fake_module.ChatOllama = fake_chat_ollama
        with patch.dict(sys.modules, {"langchain_ollama": fake_module}):
            _make_model("chosen-model", dry_run=False, max_tokens=2048)

        self.assertEqual(captured["num_predict"], 2048)


if __name__ == "__main__":
    unittest.main()
