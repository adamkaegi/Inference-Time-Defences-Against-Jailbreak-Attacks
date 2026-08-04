"""Perplexity-threshold input filter using a local causal language model."""

from dataclasses import dataclass
from math import exp, isfinite
from threading import Lock
from time import perf_counter
from typing import Any

from core import config
from .base import Defense
from .block import DefenseBlocked


@dataclass(frozen=True)
class PerplexityResult:
    """Perplexity score and token counts used to calculate it."""

    score: float | None
    token_count: int
    predicted_token_count: int
    device: str
    model_loaded_this_query: bool = False
    model_load_latency_ms: float = 0


class HuggingFacePerplexityScorer:
    """Lazily loaded, reusable causal-language-model perplexity scorer."""

    def __init__(self, model_name: str = "gpt2", device: str = "auto", stride: int = 256):
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if stride <= 0:
            raise ValueError("stride must be greater than zero")
        self.model_name = model_name
        self.requested_device = device
        self.stride = stride
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._resolved_device = None
        self._load_lock = Lock()

    @property
    def resolved_device(self) -> str:
        if self._resolved_device is None:
            self._load()
        return self._resolved_device

    def _load(self) -> tuple[bool, float]:
        if self._model is not None:
            return False, 0

        with self._load_lock:
            if self._model is not None:
                return False, 0

            start = perf_counter()
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self._resolve_device(torch, self.requested_device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            model.to(device)
            model.eval()

            # Publish the model last so concurrent callers cannot observe a
            # partially initialized scorer.
            self._torch = torch
            self._tokenizer = tokenizer
            self._resolved_device = device
            self._model = model
            return True, (perf_counter() - start) * 1000

    @staticmethod
    def _resolve_device(torch_module: Any, requested: str) -> str:
        if requested == "auto":
            if torch_module.cuda.is_available():
                return "cuda"
            mps = getattr(torch_module.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("PERPLEXITY_DEVICE is 'cuda', but CUDA is unavailable")
        if requested == "mps":
            mps = getattr(torch_module.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError("PERPLEXITY_DEVICE is 'mps', but MPS is unavailable")
        return requested

    def score(self, text: str) -> PerplexityResult:
        """Calculate aggregate strided causal perplexity without truncation."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        loaded_this_query, load_latency_ms = self._load()
        encoded = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self._resolved_device)
        token_count = int(input_ids.size(1))
        if token_count <= 1:
            return PerplexityResult(
                None,
                token_count,
                0,
                self._resolved_device,
                loaded_this_query,
                load_latency_ms,
            )

        context_length = self._context_length()
        if self.stride > context_length - 1:
            raise ValueError(
                f"stride {self.stride} exceeds the maximum {context_length - 1} "
                f"for {self.model_name}'s context length {context_length}"
            )
        total_negative_log_likelihood = 0.0
        predicted_token_count = 0
        previous_end = 0

        with self._torch.inference_mode():
            for begin in range(0, token_count, self.stride):
                end = min(begin + context_length, token_count)
                new_token_count = end - previous_end
                window_ids = input_ids[:, begin:end]
                labels = window_ids.clone()
                labels[:, :-new_token_count] = -100

                evaluated = int((labels[:, 1:] != -100).sum().item())
                if evaluated:
                    output = self._model(input_ids=window_ids, labels=labels)
                    loss = float(output.loss.item())
                    if not isfinite(loss):
                        raise ValueError("causal language model returned a non-finite loss")
                    total_negative_log_likelihood += loss * evaluated
                    predicted_token_count += evaluated

                previous_end = end
                if end == token_count:
                    break

        if predicted_token_count == 0:
            return PerplexityResult(
                None,
                token_count,
                0,
                self._resolved_device,
                loaded_this_query,
                load_latency_ms,
            )
        score = exp(total_negative_log_likelihood / predicted_token_count)
        if not isfinite(score):
            raise ValueError("calculated perplexity is not finite")
        return PerplexityResult(
            score,
            token_count,
            predicted_token_count,
            self._resolved_device,
            loaded_this_query,
            load_latency_ms,
        )

    def _context_length(self) -> int:
        candidates = [
            getattr(self._model.config, "max_position_embeddings", None),
            getattr(self._model.config, "n_positions", None),
            getattr(self._tokenizer, "model_max_length", None),
        ]
        usable = [int(value) for value in candidates if value and int(value) < 1_000_000]
        if not usable:
            raise ValueError(f"Cannot determine context length for {self.model_name}")
        context_length = min(usable)
        if context_length < 2:
            raise ValueError("causal language model context length must be at least two")
        return context_length


class PerplexityDefense(Defense):
    """Block input text whose perplexity is above the configured threshold."""

    name = "perplexity"
    stage = "input"

    def __init__(
        self,
        scorer=None,
        threshold: float = config.PERPLEXITY_THRESHOLD,
        failure_policy: str = config.PERPLEXITY_FAILURE_POLICY,
        blocked_response: str = config.PERPLEXITY_BLOCKED_RESPONSE,
    ) -> None:
        if failure_policy not in {"allow", "block", "raise"}:
            raise ValueError("failure_policy must be one of: allow, block, raise")
        self.threshold = threshold
        self.failure_policy = failure_policy
        self.blocked_response = blocked_response
        self.scorer = scorer or HuggingFacePerplexityScorer(
            model_name=config.PERPLEXITY_MODEL,
            device=config.PERPLEXITY_DEVICE,
            stride=config.PERPLEXITY_STRIDE,
        )

    def for_run(self, model_name: str, dry_run: bool = False) -> "PerplexityDefense":
        """Avoid loading GPT-2 during an offline pipeline wiring check."""
        if not dry_run:
            return self
        return PerplexityDefense(
            scorer=_DryRunPerplexityScorer(),
            threshold=self.threshold,
            failure_policy=self.failure_policy,
            blocked_response=self.blocked_response,
        )

    def apply(self, text: str) -> str:
        start = perf_counter()
        result = None
        error = ""
        status = "scored" if text else "empty_input"
        try:
            result = self.scorer.score(text)
            if result.predicted_token_count == 0 and status == "scored":
                status = "insufficient_tokens"
        except Exception as exc:
            status = "error"
            if self.failure_policy == "raise":
                raise RuntimeError(f"Perplexity input defense failed: {exc}") from exc
            error = f"{type(exc).__name__}: {exc}"

        latency_s = perf_counter() - start
        if result is not None:
            blocked = result.score is not None and result.score > self.threshold
        else:
            blocked = self.failure_policy == "block"

        metadata = {
            "stage": self.stage,
            "blocked": blocked,
            "status": status,
            "score": result.score if result is not None else None,
            "threshold": self.threshold,
            "token_count": result.token_count if result is not None else 0,
            "predicted_token_count": result.predicted_token_count if result is not None else 0,
            "model": self.scorer.model_name,
            "device": (
                result.device
                if result is not None
                else getattr(self.scorer, "requested_device", "unknown")
            ),
            "stride": self.scorer.stride,
            "latency_seconds": latency_s,
            "model_loaded_this_query": (
                result.model_loaded_this_query if result is not None else False
            ),
            "model_load_latency_ms": (
                result.model_load_latency_ms if result is not None else 0
            ),
            "failure_policy": self.failure_policy,
            "error": error,
        }
        if blocked:
            raise DefenseBlocked(
                self.blocked_response,
                defense_name=self.name,
                metadata=metadata,
            )
        return text


class _DryRunPerplexityScorer:
    model_name = "dry-run"
    requested_device = "dry-run"
    stride = config.PERPLEXITY_STRIDE

    def score(self, text: str) -> PerplexityResult:
        return PerplexityResult(None, 0, 0, "dry-run")
