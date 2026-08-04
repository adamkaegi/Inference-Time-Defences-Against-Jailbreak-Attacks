import tempfile
import unittest
from pathlib import Path

from core.attack_cache import AttackOutputCache
from main import _apply_attack


class _CountingPair:
    name = "pair"

    def __init__(self) -> None:
        self.calls = 0

    def apply(self, prompt: str) -> str:
        self.calls += 1
        return f"{prompt}|pair-{self.calls}"


class AttackOutputCacheTests(unittest.TestCase):
    def _cache(self, path: Path, *, rounds: int = 5) -> AttackOutputCache:
        return AttackOutputCache(
            path,
            attack="pair",
            model="target",
            batch="harmful",
            settings={"rounds": rounds, "seed": 42},
        )

    def test_persistent_hit_reuses_the_exact_attack_without_reapplying_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            first_attack = _CountingPair()
            first_output, _, first_hit, _ = _apply_attack(
                first_attack, "objective", self._cache(path)
            )

            second_attack = _CountingPair()
            second_output, _, second_hit, original_latency = _apply_attack(
                second_attack, "objective", self._cache(path)
            )

            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(first_output, second_output)
            self.assertEqual(first_attack.calls, 1)
            self.assertEqual(second_attack.calls, 0)
            self.assertGreaterEqual(original_latency, 0.0)

    def test_changed_attack_settings_reject_an_existing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            cache = self._cache(path, rounds=3)
            cache.put("objective", "candidate", 1.5)

            with self.assertRaisesRegex(ValueError, "settings"):
                self._cache(path, rounds=5)

    def test_existing_entry_cannot_be_silently_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = self._cache(Path(directory) / "pair.json")
            cache.put("objective", "candidate one", 1.5)

            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                cache.put("objective", "candidate two", 2.0)


if __name__ == "__main__":
    unittest.main()
