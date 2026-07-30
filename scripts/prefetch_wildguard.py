"""Download and smoke-test the pinned WildGuard refusal judge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from judges.wildguard import WildGuardRefusalJudge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow the approximately 14.5 GB classifier to run without CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA-enabled PyTorch is required for a practical WildGuard run. "
            "Install requirements-cuda.txt, or pass --allow-cpu deliberately."
        )

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"WildGuard evaluator device: {device}")

    judge = WildGuardRefusalJudge(batch_size=1, allow_cpu=args.allow_cpu)
    try:
        refusal = judge.evaluate(
            "Provide instructions for an explicitly harmful act.",
            "I cannot help with that request.",
        )
        compliance = judge.evaluate(
            "What is the capital of Canada?",
            "The capital of Canada is Ottawa.",
        )
    except RuntimeError as exc:
        raise SystemExit(f"WildGuard setup failed: {exc}") from None
    finally:
        judge.unload_model()

    print(
        "WildGuard is ready "
        f"(refusal={refusal.label}, benign compliance={compliance.label})."
    )


if __name__ == "__main__":
    main()
