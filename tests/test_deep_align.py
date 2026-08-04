import unittest

import config
from defenses.base import Defense
from defenses.deep_align import (
    DEFAULT_CONTRASTIVE_PAIRS,
    DeepAlignDefense,
    HFSteeringEngine,
)
from pipeline import build_response_chain


class _FakeLayers(list):
    """Stands in for `model.model.layers` without loading a real model."""


class _FakeInnerModel:
    def __init__(self, num_layers: int) -> None:
        self.layers = _FakeLayers(range(num_layers))


class _FakeModel:
    def __init__(self, num_layers: int) -> None:
        self.model = _FakeInnerModel(num_layers)


class _FakeEngine:
    def __init__(self, response: str = "steered response") -> None:
        self.response = response
        self.calls: list[tuple[str, int]] = []

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        self.calls.append((prompt, max_new_tokens))
        return self.response


class _SuffixOutputDefense(Defense):
    name = "suffix"
    stage = "output"

    def apply(self, text: str) -> str:
        return f"{text}|checked"


class DeepAlignContrastivePairsTests(unittest.TestCase):
    def test_default_pairs_are_well_formed(self) -> None:
        self.assertGreater(len(DEFAULT_CONTRASTIVE_PAIRS), 1)
        for pair in DEFAULT_CONTRASTIVE_PAIRS:
            self.assertTrue(pair.query.strip())
            self.assertTrue(pair.harmful_continuation.strip())
            self.assertTrue(pair.safe_continuation.strip())


class HFSteeringEngineLayerSelectionTests(unittest.TestCase):
    def test_selects_a_middle_to_late_layer_band(self) -> None:
        engine = HFSteeringEngine(
            model_name="unused",
            layer_fraction_start=0.6,
            layer_fraction_end=0.95,
        )
        engine._model = _FakeModel(num_layers=32)

        layers = engine._select_layers()

        self.assertEqual(layers, list(range(19, 30)))

    def test_rejects_invalid_layer_fractions(self) -> None:
        with self.assertRaises(ValueError):
            HFSteeringEngine(model_name="unused", layer_fraction_start=0.9, layer_fraction_end=0.5)

    def test_rejects_non_positive_context_tokens(self) -> None:
        with self.assertRaises(ValueError):
            HFSteeringEngine(model_name="unused", context_tokens=0)

    def test_rejects_empty_contrastive_pairs(self) -> None:
        with self.assertRaises(ValueError):
            HFSteeringEngine(model_name="unused", contrastive_pairs=())

    def test_finds_llama_style_decoder_layers(self) -> None:
        engine = HFSteeringEngine(model_name="unused")
        engine._model = _FakeModel(num_layers=4)

        self.assertIs(engine._decoder_layers(), engine._model.model.layers)


class DeepAlignDefenseTests(unittest.TestCase):
    def test_apply_delegates_to_the_injected_engine(self) -> None:
        fake_engine = _FakeEngine()
        defense = DeepAlignDefense(engine=fake_engine, max_new_tokens=123)

        result = defense.apply("attacked prompt")

        self.assertEqual(result, "steered response")
        self.assertEqual(fake_engine.calls, [("attacked prompt", 123)])

    def test_for_run_maps_ollama_tag_to_hf_repo_id(self) -> None:
        defense = DeepAlignDefense()

        configured = defense.for_run("qwen2.5:7b-instruct", dry_run=True, max_tokens=64)

        self.assertEqual(
            configured.model_name, config.DEEP_ALIGN_HF_MODEL_MAP["qwen2.5:7b-instruct"]
        )
        self.assertTrue(configured.dry_run)
        self.assertEqual(configured.max_new_tokens, 64)

    def test_for_run_falls_back_to_configured_model_for_unmapped_tags(self) -> None:
        defense = DeepAlignDefense(model_name="explicit/override")

        configured = defense.for_run("some-unmapped-tag", dry_run=True)

        self.assertEqual(configured.model_name, "explicit/override")

    def test_dry_run_engine_echoes_the_prompt_without_loading_a_model(self) -> None:
        defense = DeepAlignDefense().for_run("qwen2.5:7b-instruct", dry_run=True)

        result = defense.apply("hello")

        self.assertIn("hello", result)

    def test_pipeline_does_not_make_an_extra_outer_model_call(self) -> None:
        from unittest.mock import patch

        fake_engine = _FakeEngine(response="response")
        deep_align = DeepAlignDefense(engine=fake_engine)

        with patch("pipeline._make_model", side_effect=AssertionError("outer call")):
            chain = build_response_chain(
                [deep_align, _SuffixOutputDefense()],
                "unused-outer-model",
                dry_run=True,
            )
            result = chain.invoke("prompt")

        self.assertEqual(result, "response|checked")
        self.assertEqual(len(fake_engine.calls), 1)


if __name__ == "__main__":
    unittest.main()
