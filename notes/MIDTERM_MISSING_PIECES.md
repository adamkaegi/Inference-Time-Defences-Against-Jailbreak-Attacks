# Midterm Missing Pieces

- Attack implementation gaps
  - The current `gcg` attack is a static adversarial suffix baseline, not a complete implementation of gradient-based Greedy Coordinate Gradient optimization.
  - GCG is not optimized specifically for the Qwen 2.5 target model.
  - Prompt-specific GCG artifacts have been downloaded, but the active GCG implementation does not load or select among them.
  - The current `pair` attack is a simplified PAIR-style prototype rather than a faithful reproduction of the published algorithm.
  - PAIR uses a lightweight keyword-based stopping oracle instead of a goal-aware compliance judge.
  - PAIR does not currently verify that rewritten prompts preserve the original harmful objective.
  - Some PAIR candidates weaken, reverse, or otherwise change the original objective.
  - The configured multi-stream PAIR search is not connected to the active implementation.
  - PAIR does not retain and compare the best candidate from all rounds and streams.
  - PAIR does not provide detailed judge scores and improvement feedback to the attacker after every round.
  - Attack query counts, selected-round information, compliance scores, and fidelity scores are not recorded in the final result rows.
  - DeepInception is the strongest implemented semantic attack, but its long nested responses substantially increase generation time.
  - A concise DeepInception variant could be explored while keeping the published five-layer version as the primary reproducible baseline.

- Attack consistency and caching status
  - PAIR attack prompts are now generated once per target model, attack configuration, prompt batch, and original prompt.
  - Each defense cell receives the exact same cached PAIR prompt, supporting a controlled oblivious-threat-model comparison.
  - Adaptive attacks against a defended pipeline should be reported separately as stress tests.
  - `attack_cache.py` is integrated into `main.py`; matrix runs automatically share a validated persistent PAIR cache across defense-cell subprocesses.
  - The cache identity covers the active PAIR settings, target model, attacker system prompt, stopping judge, seed, and prompt-batch hash.
  - Cache-hit status and PAIR query counts are not yet stored as dedicated final-result columns.

- Defense scope gaps
  - `llama_guard_input` remains available for ad-hoc experiments but has been removed from the default matrix.
  - The proposal-aligned defense stack is `perplexity,smoothllm,llama_guard_output`.
  - The current SmoothLLM implementation uses five perturbation samples to control runtime; a larger sample count may provide a stronger defense but is more expensive.
  - SmoothLLM uses a lightweight refusal-language predicate for its internal vote rather than a heavyweight behavior judge.
  - SmoothLLM hyperparameter sensitivity has not yet been evaluated.
  - Perplexity threshold sensitivity has not yet been evaluated on a held-out calibration set.
  - Defense ordering has not been systematically compared beyond the selected proposal-aligned stack.
  - The project does not yet include adaptive attacks designed specifically to bypass each defense.

- Judge and metric limitations
  - HarmBench behavior ASR should be treated as the primary harmful-behavior success metric.
  - JBB no-refusal rate measures refusal behavior and should not be presented as equivalent to harmful-behavior ASR.
  - The local Ollama JBB refusal judge showed substantial disagreement with HarmBench and manual refusal-marker inspection on DeepInception outputs.
  - Local JBB results should be described as exploratory until they are validated against the official provider or a human-labeled sample.
  - The exact Together-backed JBB judge is supported by the code but no Together API key is currently configured.
  - HarmBench uses the pinned Mistral validation classifier rather than HarmBench's canonical final/test classifier.
  - A small human audit should be performed on judge disagreements and apparent attack successes.
  - Inter-rater agreement or judge agreement statistics are not currently reported.
  - Confidence intervals are not currently included in the generated summary.
  - Results from the 10-prompt quick batch have high uncertainty and should be labeled preliminary.
  - Benign false-refusal metrics require the JBB refusal judge; HarmBench alone does not evaluate benign usability.

- Dataset and experimental-design limitations
  - The quick 10-prompt harmful batch is concentrated in a small subset of harmful behavior types and is not sufficient for final conclusions.
  - Presentation-grade results should use the complete 100 harmful and 100 benign JailbreakBench prompt sets.
  - Attack development and final evaluation prompts are not currently separated into explicit development and held-out test splits.
  - Attack parameters should be frozen before generating presentation-grade results.
  - Selecting attacks with the final HarmBench labels on the same evaluation prompts would introduce evaluation leakage and should be avoided.
  - The experiment currently evaluates one primary target model; cross-model transfer and sensitivity results remain future work.
  - Statistical significance tests and confidence intervals for defense comparisons remain future work.

- Runtime and efficiency limitations
  - SmoothLLM is the largest response-generation cost because it invokes the target model multiple times per prompt.
  - PAIR search is expensive because it alternates between attacker and target model calls over several rounds.
  - PAIR search is now paid once per prompt and reused across defenses instead of being repeated for every cell.
  - Ordinary generations are capped at 512 tokens to reduce runtime.
  - DeepInception generations are capped at 2048 tokens so the model has enough room to reach the final nested layer.
  - The same per-attack token limits are passed into SmoothLLM's internal generations.
  - The matrix runner processes cells sequentially and does not batch target requests across prompts.
  - Parallelizing entire cells on one GPU may create VRAM contention; limited within-stage batching should be benchmarked instead.
  - Full model outputs are printed to the terminal and log, adding noise and a small amount of overhead.
  - Atomic CSV checkpoints rewrite the growing file after each completed prompt, favoring recovery over maximum write performance.

- Reproducibility and reporting gaps
  - Final presentation runs should be generated from a clean Git commit rather than a dirty working tree.
  - The current manifest records the commit and dirty state but does not preserve a full patch of uncommitted changes.
  - The manifest should record all future PAIR stream counts, goal-judge settings, GCG artifact revisions, candidate hashes, and query budgets.
  - Old runs generated with different token limits, defense sets, attack settings, or judge sets must not be combined with new runs.
  - The previous quick run used 4096 target tokens, ten SmoothLLM samples, a three-round Qwen 3B PAIR attacker, and seven defense conditions, so it is not directly comparable with the current configuration.
  - The current default matrix contains 30 cells: 24 harmful attack-defense cells and six benign usability cells.
  - Interrupted runs can be resumed only with the same output directory and identical manifest settings.
  - The generated summary reports response-pipeline latency but should give equal prominence to attack-search latency and total latency.
  - Attack success should be reported with the exact numerator, denominator, judge, target model, prompt set, and threat model.

- Midterm presentation framing
  - Describe GCG as a static transfer-suffix or GCG-style baseline, not a complete target-optimized GCG implementation.
  - Describe PAIR as a simplified PAIR-style prototype, not a faithful reproduction of the published algorithm.
  - Present DeepInception as the strongest currently implemented attack baseline.
  - Present weak GCG and PAIR results as implementation limitations and future-work motivation rather than evidence that the attacks are intrinsically ineffective.
  - Clearly distinguish harmful-behavior ASR from no-refusal rate.
  - Label local JBB refusal results as exploratory.
  - Label 10-prompt results as preliminary and avoid strong comparative claims from them.
  - State that PAIR defense comparisons use the same validated cached attacked prompt.
  - Emphasize that checkpointing, pinned datasets, pinned HarmBench weights, deterministic target decoding, benign usability testing, and separate attack/defense latency are already implemented.
  - Identify goal-aware PAIR, prompt-specific or target-optimized GCG, judge validation, and full statistical evaluation as the main next steps.

- Recommended midterm execution scope
  - Run the complete 100-prompt matrix for `none`, `deepinception`, and the static GCG-style baseline.
  - The simplified PAIR prototype can be compared across defenses without regenerating its selected prompt, but its results must still be labeled preliminary.
  - Keep both HarmBench and JBB enabled when benign usability cells are included.
  - If HarmBench is used as the only judge, skip benign cells because they will not receive a false-refusal label.
  - Use a new output directory for the current configuration rather than resuming the earlier quick run.
  - Preserve the output directory path so an interrupted run can be resumed with identical settings.
