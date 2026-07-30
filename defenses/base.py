from abc import ABC, abstractmethod
from typing import Literal


class Defense(ABC):
    """Context-only interface template for concrete defense implementations.

    ``Defense`` itself is never registered or run. A concrete defense declares
    whether it operates on model input, generation, or model output and
    implements ``apply``.
    """

    name: str
    stage: Literal["input", "generation", "output"]

    @abstractmethod
    def apply(self, text: str) -> str:
        ...

    def apply_with_context(self, text: str, *, prompt: str) -> str:
        """Apply a defense with the model-facing prompt when it is available.

        Most defenses only need the text at their own stage. Output classifiers
        that evaluate a conversation can override this hook to inspect both the
        user prompt and assistant response without changing the simple ``apply``
        contract used by existing defenses.
        """
        return self.apply(text)
