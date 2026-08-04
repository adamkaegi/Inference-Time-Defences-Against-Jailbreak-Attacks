"""Composable Llama Guard 3 input and output filters for Ollama."""

from dataclasses import asdict, dataclass
import re
from time import perf_counter
from typing import Any

from core import config
from .base import Defense
from .block import DefenseBlocked


_VALID_FAILURE_POLICIES = {"allow", "block", "raise"}
_CATEGORY_RE = re.compile(r"\bS(?:[1-9]|1[0-3])\b", re.IGNORECASE)
_ANY_CATEGORY_RE = re.compile(r"\bS\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class GuardDecision:
    """One serializable Llama Guard classification result."""

    label: str
    categories: list[str]
    raw_output: str
    parse_status: str
    error: str
    latency_s: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def parse_guard_output(raw_output: str) -> GuardDecision:
    """Parse Llama Guard's exact safe/unsafe label and optional S1-S13 codes."""
    if not isinstance(raw_output, str):
        return _parse_error("", "Guard output must be a string")

    raw = raw_output
    stripped = raw.strip()
    if not stripped:
        return _parse_error(raw, "Guard output is empty")

    lines = stripped.splitlines()
    label = lines[0].strip().lower()
    if label not in {"safe", "unsafe"}:
        return _parse_error(raw, "First line must be exactly 'safe' or 'unsafe'")

    remainder = "\n".join(lines[1:]).strip()
    codes = [match.group(0).upper() for match in _ANY_CATEGORY_RE.finditer(remainder)]
    invalid = [code for code in codes if _CATEGORY_RE.fullmatch(code) is None]
    if invalid:
        return _parse_error(raw, f"Invalid Guard categories: {', '.join(invalid)}")
    if label == "safe" and remainder:
        return _parse_error(raw, "A safe result must not include categories or extra text")
    if label == "unsafe" and remainder:
        cleaned = _ANY_CATEGORY_RE.sub("", remainder)
        if cleaned.replace(",", "").strip():
            return _parse_error(raw, "Unsafe category lines contain unexpected text")

    categories = []
    for code in codes:
        if code not in categories:
            categories.append(code)
    return GuardDecision(label, categories, raw, "parsed", "")


def _parse_error(raw_output: str, error: str) -> GuardDecision:
    return GuardDecision("unknown", [], raw_output, "malformed", error)


def build_input_guard_conversation(text: str) -> list[dict[str, str]]:
    if not isinstance(text, str):
        raise TypeError("input text must be a string")
    return [{"role": "user", "content": text}]


def build_output_guard_conversation(
    prompt: str,
    response: str,
) -> list[dict[str, str]]:
    if not isinstance(prompt, str):
        raise TypeError("input prompt must be a string")
    if not isinstance(response, str):
        raise TypeError("target response must be a string")
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


class GuardClassifier:
    """Lazy LangChain/Ollama adapter around the tested Guard parser."""

    def __init__(self, model_name: str = config.GUARD_MODEL, client=None) -> None:
        self.model_name = model_name
        self._client = client

    def _get_client(self):
        if self._client is None:
            from langchain_ollama import ChatOllama

            self._client = ChatOllama(model=self.model_name, temperature=0)
        return self._client

    def classify_conversation(self, conversation: list[dict[str, str]]) -> GuardDecision:
        start = perf_counter()
        try:
            response = self._get_client().invoke(conversation)
            raw_output = getattr(response, "content", response)
            if not isinstance(raw_output, str):
                raw_output = str(raw_output)
        except Exception as exc:
            return GuardDecision(
                "unknown",
                [],
                "",
                "provider_error",
                f"{type(exc).__name__}: {exc}",
                latency_s=perf_counter() - start,
                model=self.model_name,
            )

        latency_s = perf_counter() - start
        parsed = parse_guard_output(raw_output)
        stats = _response_stats(response, latency_s)
        return GuardDecision(
            parsed.label,
            parsed.categories,
            parsed.raw_output,
            parsed.parse_status,
            parsed.error,
            model=self.model_name,
            **stats,
        )


class _LlamaGuardDefense(Defense):
    def __init__(
        self,
        classifier: GuardClassifier | None = None,
        failure_policy: str = config.LLAMA_GUARD_FAILURE_POLICY,
        blocked_response: str = config.LLAMA_GUARD_BLOCKED_RESPONSE,
    ) -> None:
        if failure_policy not in _VALID_FAILURE_POLICIES:
            raise ValueError("failure_policy must be one of: allow, block, raise")
        self.classifier = classifier or GuardClassifier()
        self.failure_policy = failure_policy
        self.blocked_response = blocked_response

    def for_run(self, model_name: str, dry_run: bool = False):
        """Avoid calling Ollama during an offline pipeline wiring check."""
        if not dry_run:
            return self
        return type(self)(
            classifier=_DryRunGuardClassifier(),
            failure_policy=self.failure_policy,
            blocked_response=self.blocked_response,
        )

    def _action(self, decision: GuardDecision) -> str:
        if decision.label == "unsafe":
            return "block"
        if decision.label == "safe":
            return "allow"
        if self.failure_policy == "raise":
            detail = decision.error or decision.parse_status
            raise RuntimeError(f"Llama Guard {self.stage} classification failed: {detail}")
        return self.failure_policy

    def _apply_decision(
        self,
        text: str,
        decision: GuardDecision,
    ) -> str:
        blocked = self._action(decision) == "block"
        metadata = {
            "stage": self.stage,
            "blocked": blocked,
            "response_replaced": blocked and self.stage == "output",
            "guard_model": self.classifier.model_name,
            "failure_policy": self.failure_policy,
            "decision": decision.to_dict(),
        }
        if not blocked:
            return text
        if self.stage == "input":
            raise DefenseBlocked(
                self.blocked_response,
                defense_name=self.name,
                metadata=metadata,
            )
        return self.blocked_response


class LlamaGuardInputDefense(_LlamaGuardDefense):
    """Classify model-facing input and short-circuit unsafe requests."""

    name = "llama_guard_input"
    stage = "input"

    def apply(self, text: str) -> str:
        conversation = build_input_guard_conversation(text)
        decision = self.classifier.classify_conversation(conversation)
        return self._apply_decision(text, decision)


class LlamaGuardOutputDefense(_LlamaGuardDefense):
    """Classify and replace unsafe target responses."""

    name = "llama_guard_output"
    stage = "output"

    def apply(self, text: str) -> str:
        # Keep the plain Defense interface usable for direct callers. The
        # response pipeline calls apply_with_context so real runs classify the
        # complete user/assistant exchange.
        conversation = [{"role": "assistant", "content": text}]
        decision = self.classifier.classify_conversation(conversation)
        return self._apply_decision(text, decision)

    def apply_with_context(self, text: str, *, prompt: str) -> str:
        conversation = build_output_guard_conversation(prompt, text)
        decision = self.classifier.classify_conversation(conversation)
        return self._apply_decision(text, decision)


class _DryRunGuardClassifier:
    model_name = "dry-run"

    def classify_conversation(self, conversation: list[dict[str, str]]) -> GuardDecision:
        return GuardDecision("safe", [], "safe", "dry_run", "", model=self.model_name)


def _response_stats(response: Any, latency_s: float) -> dict[str, float | int]:
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = getattr(response, "usage_metadata", {}) or {}
    prompt_tokens = int(
        metadata.get("prompt_eval_count", usage.get("input_tokens", 0)) or 0
    )
    completion_tokens = int(
        metadata.get("eval_count", usage.get("output_tokens", 0)) or 0
    )
    total_tokens = int(
        usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
    )
    eval_duration_ns = metadata.get("eval_duration", 0) or 0
    eval_duration_s = eval_duration_ns / 1_000_000_000 if eval_duration_ns else 0
    return {
        "latency_s": latency_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": (
            completion_tokens / eval_duration_s if eval_duration_s else 0
        ),
    }
