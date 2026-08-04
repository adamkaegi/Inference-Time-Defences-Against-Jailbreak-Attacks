"""Builds the LangChain pipeline that turns one prompt into one model response.

LangChain's "LCEL" (LangChain Expression Language) composes steps with the `|`
operator, like piping shell commands. Every step is a `Runnable` — an object
with `.invoke(x)` that takes one value and returns the next one. `a | b`
gives you a new Runnable that feeds `a`'s output straight into `b`.

Our chain, left to right:

    attack -> [input-stage defenses] -> prompt template -> model -> text parser -> [output-stage defenses]

- `attack.apply` and `defense.apply` are plain Python functions (see
  attacks/base.py, defenses/base.py). Wrapping them in `RunnableLambda` is
  what lets them sit in the `|` chain alongside LangChain's own components.
- `ChatPromptTemplate` turns `{"text": "..."}` into the list of chat messages
  the model expects.
- `ChatOllama` is the actual LLM call.
- `StrOutputParser` unwraps the model's reply object down to a plain string.

Each defense declares a `stage` ("input" or "output"), so adding or removing
a defense never requires touching this file — it just gets spliced in at the
matching point.
"""

from collections.abc import Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

import config
from attacks.base import Attack
from defenses.base import Defense


def _make_model(
    model_name: str,
    dry_run: bool,
    max_tokens: int | None = None,
) -> Runnable:
    """Return the real Ollama model, or a stub that just echoes the prompt.

    --dry-run lets the rest of the chain be exercised without an Ollama
    server running.
    """
    if dry_run:
        return RunnableLambda(
            lambda prompt_value: f"[dry-run model] prompt was:\n{prompt_value.to_string()}"
        )

    from langchain_ollama import ChatOllama  # imported lazily: not needed for --dry-run

    resolved_max_tokens = config.MODEL_MAX_TOKENS if max_tokens is None else max_tokens
    return ChatOllama(
        model=model_name,
        temperature=config.MODEL_TEMPERATURE,
        seed=config.MODEL_SEED,
        num_predict=resolved_max_tokens,
    )


def _defense_step(defense: Defense) -> Runnable:
    """Wrap a Defense as a Runnable step in the chain.

    `defense=defense` captures the current loop value immediately. Without
    it, every step's lambda would share one `defense` variable and all of
    them would run the *last* defense in the list once the loop finished.
    """
    return RunnableLambda(lambda text, defense=defense: defense.apply(text))


def _output_defense_step(defense: Defense) -> Runnable:
    """Apply an output defense while retaining its model-facing prompt."""
    return RunnableLambda(
        lambda state, defense=defense: {
            "prompt": state["prompt"],
            "response": defense.apply_with_context(
                state["response"],
                prompt=state["prompt"],
            ),
        }
    )


def build_response_chain(
    defenses: Sequence[Defense],
    model_name: str,
    dry_run: bool = False,
    max_tokens: int | None = None,
) -> Runnable:
    """Wire defenses and a model after an attack has already been applied.

    Call ``.invoke(attacked_prompt)`` on the result. This lets callers cache
    the exact attack output without applying stateful or stochastic attacks
    twice.
    """
    input_defenses = [d for d in defenses if d.stage == "input"]
    generation_defenses = [d for d in defenses if d.stage == "generation"]
    output_defenses = [d for d in defenses if d.stage == "output"]

    if len(generation_defenses) > 1:
        names = ", ".join(defense.name for defense in generation_defenses)
        raise ValueError(f"Only one generation-stage defense is supported: {names}")

    prompt_template = ChatPromptTemplate.from_messages([("user", "{text}")])

    chain = RunnableLambda(lambda text: text)
    for defense in input_defenses:
        chain = chain | _defense_step(defense)

    if generation_defenses:
        # A generation-stage defense owns the target calls and returns the
        # selected assistant response. It replaces the ordinary one-call model
        # stage, while output defenses still run afterward.
        generation_chain = _defense_step(generation_defenses[0])
    else:
        model = _make_model(model_name, dry_run, max_tokens=max_tokens)
        # ChatPromptTemplate needs a dict of template variables, so re-wrap the
        # plain string as {"text": ...} before it reaches the template.
        generation_chain = (
            RunnableLambda(lambda text: {"text": text})
            | prompt_template
            | model
            | StrOutputParser()
        )

    # Retain the exact post-input-defense prompt alongside the generated text.
    # Output defenses such as Llama Guard need the full user/assistant exchange
    # to classify a response according to the guard model's chat template.
    chain = (
        chain
        | RunnableLambda(lambda prompt: {"prompt": prompt})
        | RunnablePassthrough.assign(
            response=RunnableLambda(lambda state: state["prompt"])
            | generation_chain
        )
    )

    for defense in output_defenses:
        chain = chain | _output_defense_step(defense)

    return chain | RunnableLambda(lambda state: state["response"])


def build_chain(
    attack: Attack,
    defenses: Sequence[Defense],
    model_name: str,
    dry_run: bool = False,
) -> Runnable:
    """Wire attack, defenses, and a model into the original public Runnable."""
    return RunnableLambda(attack.apply) | build_response_chain(
        defenses,
        model_name,
        dry_run=dry_run,
        max_tokens=config.model_max_tokens_for_attack(attack.name),
    )
