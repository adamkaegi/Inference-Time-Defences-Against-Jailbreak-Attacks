"""Atomic, validated cache for attack outputs shared across defense cells."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


SCHEMA_VERSION = 2


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def settings_fingerprint(settings: Mapping[str, Any]) -> str:
    """Return a stable identity for every setting that can change an attack."""
    encoded = json.dumps(
        settings,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CachedAttackOutput:
    attacked_prompt: str
    generation_latency_seconds: float


class AttackOutputCache:
    """One model/attack/batch cache, written after every generated attack."""

    def __init__(
        self,
        path: Path,
        *,
        attack: str,
        model: str,
        batch: str,
        settings: Mapping[str, Any],
    ) -> None:
        self.path = path.resolve()
        self.attack = attack
        self.model = model
        self.batch = batch
        # Round-trip through JSON so the value written and the value validated
        # have exactly the same representation.
        self.settings = json.loads(
            json.dumps(settings, ensure_ascii=False, sort_keys=True)
        )
        self.settings_sha256 = settings_fingerprint(self.settings)
        self._entries: dict[str, dict[str, object]] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": SCHEMA_VERSION,
            "attack": self.attack,
            "model": self.model,
            "batch": self.batch,
            "settings_sha256": self.settings_sha256,
        }
        for field, value in expected.items():
            if payload.get(field) != value:
                raise ValueError(
                    f"Attack cache {self.path} has {field}={payload.get(field)!r}; "
                    f"expected {value!r}"
                )
        if payload.get("settings") != self.settings:
            raise ValueError(f"Attack cache {self.path} has different attack settings")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"Attack cache {self.path} has invalid entries")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Attack cache {self.path} contains a malformed entry")
            prompt = entry.get("input")
            attacked = entry.get("attacked_input")
            if not isinstance(prompt, str) or not isinstance(attacked, str):
                raise ValueError(f"Attack cache {self.path} contains non-text prompts")
            key = _prompt_hash(prompt)
            if entry.get("prompt_sha256") != key or key in self._entries:
                raise ValueError(
                    f"Attack cache {self.path} has a duplicate or invalid prompt hash"
                )
            self._entries[key] = dict(entry)

    def get(self, prompt: str) -> CachedAttackOutput | None:
        entry = self._entries.get(_prompt_hash(prompt))
        if entry is None:
            return None
        if entry.get("input") != prompt:
            raise ValueError("attack cache hash collision or corrupted prompt entry")
        return CachedAttackOutput(
            attacked_prompt=str(entry["attacked_input"]),
            generation_latency_seconds=float(
                entry.get("generation_latency_seconds", 0.0)
            ),
        )

    def put(
        self,
        prompt: str,
        attacked_prompt: str,
        generation_latency_seconds: float,
    ) -> None:
        existing = self.get(prompt)
        if existing is not None:
            if existing.attacked_prompt != attacked_prompt:
                raise ValueError(
                    "Refusing to overwrite a cached prompt with a different attack output"
                )
            return
        key = _prompt_hash(prompt)
        self._entries[key] = {
            "prompt_sha256": key,
            "input": prompt,
            "attacked_input": attacked_prompt,
            "generation_latency_seconds": float(generation_latency_seconds),
        }
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "attack": self.attack,
            "model": self.model,
            "batch": self.batch,
            "settings": self.settings,
            "settings_sha256": self.settings_sha256,
            "entries": list(self._entries.values()),
        }
        temporary_path = self.path.with_name(
            f".{self.path.name}.{uuid4().hex[:8]}.tmp"
        )
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)
