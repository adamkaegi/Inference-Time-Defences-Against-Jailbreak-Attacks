"""SmoothLLM randomized smoothing as a generation-stage defense."""

from __future__ import annotations

import random
import string
from collections.abc import Callable

from core import config
from judges.sample_safe_unsafe import looks_like_refusal
from .base import Defense


class SmoothLLMDefense(Defense):
    """Generate perturbed responses and return one from the majority class.

    The perturbation is random character replacement. Responses are grouped
    with a lightweight refusal-language predicate, matching the proof-of-
    concept metric used by the project; the final HarmBench/JBB judges remain
    responsible for reported experiment metrics.
    """

    name = "smoothllm"
    stage = "generation"

    def __init__(
        self,
        model_name: str = config.MODEL,
        num_samples: int = config.SMOOTHLLM_NUM_SAMPLES,
        perturbation_rate: float = config.SMOOTHLLM_PERTURBATION_RATE,
        max_tokens: int = config.MODEL_MAX_TOKENS,
        seed: int | None = config.MODEL_SEED,
        dry_run: bool = False,
        generator: Callable[[str], str] | None = None,
        jailbreak_classifier: Callable[[str], bool] | None = None,
    ) -> None:
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if not 0.0 <= perturbation_rate <= 1.0:
            raise ValueError("perturbation_rate must be between 0.0 and 1.0")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        self.model_name = model_name
        self.num_samples = num_samples
        self.perturbation_rate = perturbation_rate
        self.max_tokens = max_tokens
        self.seed = seed
        self.dry_run = dry_run
        self._injected_generator = generator
        self._jailbreak_classifier = (
            jailbreak_classifier or self._looks_jailbroken
        )
        self._model = None

    def apply(self, text: str) -> str:
        candidates = [self._perturb(text, index) for index in range(self.num_samples)]
        responses = [self._generate(candidate) for candidate in candidates]
        verdicts = [self._jailbreak_classifier(response) for response in responses]

        jailbreak_responses = [
            response
            for response, jailbroken in zip(responses, verdicts)
            if jailbroken
        ]
        refusal_responses = [
            response
            for response, jailbroken in zip(responses, verdicts)
            if not jailbroken
        ]

        # Match SmoothLLM's strict-majority rule. A tie selects the
        # non-jailbroken/refusal class.
        majority_responses = (
            jailbreak_responses
            if len(jailbreak_responses) > self.num_samples / 2
            else refusal_responses
        )
        if not majority_responses:
            # This is only reachable with a broken classifier contract, but a
            # deterministic response is safer than performing an N+1th model call.
            majority_responses = responses
        return self._choice(majority_responses)

    def for_model(self, model_name: str) -> "SmoothLLMDefense":
        return type(self)(
            model_name=model_name,
            num_samples=self.num_samples,
            perturbation_rate=self.perturbation_rate,
            max_tokens=self.max_tokens,
            seed=self.seed,
            dry_run=self.dry_run,
        )

    def for_run(
        self,
        model_name: str,
        dry_run: bool = False,
        max_tokens: int | None = None,
    ) -> "SmoothLLMDefense":
        return type(self)(
            model_name=model_name,
            num_samples=self.num_samples,
            perturbation_rate=self.perturbation_rate,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            seed=self.seed,
            dry_run=dry_run,
        )

    def _choice(self, responses: list[str]) -> str:
        if not responses:
            raise RuntimeError("SmoothLLM produced no candidate responses")
        if self.seed is None:
            return random.choice(responses)
        return random.Random(self.seed).choice(responses)

    def _perturb(self, text: str, sample_index: int) -> str:
        rng = (
            random.Random()
            if self.seed is None
            else random.Random(self.seed + sample_index * 7919)
        )
        alphabet = string.ascii_letters + string.digits + string.punctuation
        perturbed: list[str] = []
        for char in text:
            if char.isspace() or rng.random() >= self.perturbation_rate:
                perturbed.append(char)
                continue
            replacements = alphabet.replace(char, "") or alphabet
            perturbed.append(rng.choice(replacements))
        return "".join(perturbed)

    def _generate(self, prompt: str) -> str:
        if self._injected_generator is not None:
            response = self._injected_generator(prompt)
            if not isinstance(response, str):
                raise TypeError("SmoothLLM generator must return a string")
            return response
        if self.dry_run:
            return f"[dry-run smoothllm model] prompt was:\n{prompt}"

        try:
            response = self._get_model().invoke(prompt)
        except Exception as exc:
            raise RuntimeError("SmoothLLM target generation failed") from exc
        return self._content(response)

    def _get_model(self):
        if self._model is None:
            from langchain_ollama import ChatOllama

            self._model = ChatOllama(
                model=self.model_name,
                temperature=config.MODEL_TEMPERATURE,
                seed=config.MODEL_SEED,
                num_predict=self.max_tokens,
            )
        return self._model

    @staticmethod
    def _looks_jailbroken(response: str) -> bool:
        """Return a documented no-refusal proxy for the internal majority vote."""
        return not looks_like_refusal(response)

    @staticmethod
    def _content(response: object) -> str:
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)
