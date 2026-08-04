"""Integration tests for block-aware defense composition."""

import pytest

from attacks.base import Attack
from defenses.base import Defense
from defenses.block import DefenseBlocked
from main import _invoke_chain
from core.pipeline import build_chain


class IdentityAttack(Attack):
    name = "identity"

    def apply(self, text: str) -> str:
        return text


class LegacyTextDefense(Defense):
    name = "legacy"
    stage = "input"

    def apply(self, text: str) -> str:
        return text + " transformed"


class BlockingDefense(Defense):
    name = "blocker"
    stage = "input"

    def apply(self, text: str) -> str:
        raise DefenseBlocked("blocked response", defense_name=self.name)


class OutputDefense(Defense):
    name = "output"
    stage = "output"

    def apply(self, text: str) -> str:
        return text + " output-checked"


def test_legacy_string_defense_remains_pluggable():
    chain = build_chain(IdentityAttack(), [LegacyTextDefense()], "unused", dry_run=True)

    output = chain.invoke("hello")

    assert "hello transformed" in output


def test_input_block_signal_stops_the_pipeline():
    chain = build_chain(
        IdentityAttack(),
        [BlockingDefense(), OutputDefense()],
        "unused",
        dry_run=True,
    )

    with pytest.raises(DefenseBlocked) as blocked:
        chain.invoke("hello")

    assert blocked.value.response == "blocked response"


def test_main_adapter_returns_blocked_response():
    chain = build_chain(
        IdentityAttack(),
        [BlockingDefense(), OutputDefense()],
        "unused",
        dry_run=True,
    )

    assert _invoke_chain(chain, "hello", {}) == "blocked response"


def test_output_defense_remains_a_plain_text_transform():
    chain = build_chain(
        IdentityAttack(),
        [OutputDefense()],
        "unused",
        dry_run=True,
    )

    output = chain.invoke("hello")

    assert output.endswith(" output-checked")
