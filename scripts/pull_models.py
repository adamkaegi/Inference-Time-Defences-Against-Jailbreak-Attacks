"""Pull every Ollama target/support model used by the project."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core import config  # noqa: E402  (project root is added above)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets-only",
        action="store_true",
        help="Skip support models such as Llama Guard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = list(config.TARGET_MODELS)
    if not args.targets_only:
        models.extend(config.SUPPORT_MODELS)

    for index, model in enumerate(models, 1):
        print(f"[{index}/{len(models)}] pulling {model}", flush=True)
        subprocess.run(["ollama", "pull", model], check=True)

    print("\nAll requested Ollama models are available locally.")


if __name__ == "__main__":
    main()
