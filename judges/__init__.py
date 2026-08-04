import config

from .base import Judge
from .harmbench_mistral import HarmBenchMistralJudge
from .jailbreak_bench_llama8b import JailbreakBenchRefusalJudge
from .sample_safe_unsafe import SampleSafeUnsafeJudge
from .strongreject import StrongRejectJudge
from .wildguard import WildGuardRefusalJudge

# Available judges:
# - sample_safe_unsafe: keyword-only wiring/debug judge.
# - strongreject: continuous 0..1 harmful-assistance evaluator.
# - harmbench_mistral_7b_val_cls: official binary HarmBench validation classifier.
# - wildguard_refusal: WildGuard refusal/compliance classifier.
# - jbb_refusal_llama3_8b: JBB Llama 3 8B refusal classifier.
# - jailbreak_bench_llama8b: compatibility alias for jbb_refusal_llama3_8b.
_jbb_refusal = JailbreakBenchRefusalJudge(provider=config.JBB_REFUSAL_PROVIDER)

JUDGES: dict[str, Judge] = {
    judge.name: judge
    for judge in (
        SampleSafeUnsafeJudge(),
        StrongRejectJudge(),
        HarmBenchMistralJudge(),
        WildGuardRefusalJudge(),
        _jbb_refusal,
    )
}

JUDGES["jailbreak_bench_llama8b"] = _jbb_refusal
