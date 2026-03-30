# HANDOFF: Entropy-Based Confabulation Detection ("Conscience in the Engine")

## Concept

Instrument vllm-mlx's inference engine to detect high entropy in per-token logit distributions during generation. When entropy exceeds a threshold (model is uncertain), inject thinking tokens into the generation stream:

```
<think>I'm uncertain about this claim. I should verify before asserting.</think>
```

The model then conditions subsequent tokens on that injected thought, nudging it toward uncertainty-aware reasoning instead of confident confabulation.

**This is only possible because we own the inference engine.** Can't do this through an API.

## Origin

Stuart's idea, communicated via Cyril (2026-03-30). References:
- Farquhar et al., "Detecting hallucinations in large language models using semantic entropy" (Nature, 2024): https://www.nature.com/articles/s41586-024-07421-0
- HalluField framework: operates directly on logits with temperature perturbation

## Key Technical Questions

### 1. Can vllm-mlx expose per-token logit entropy?

**Answer: Yes, straightforward.**

The logits are computed at each generation step in the scheduler. The sampling step has full access to the logit tensor before sampling a token.

Entropy computation: `H = -sum(p * log(p))` where `p = softmax(logits)`.

**Where to tap in:**
- `vllm_mlx/scheduler.py` - the `_generation_step()` method, after logits are computed but before sampling
- Or in the sampler itself if vllm-mlx uses a separate sampling module

**What to look at:**
- `scheduler.py` line ~2179: `responses = self.batch_generator.next()` - this is where tokens are generated
- The `BatchGenerator` from mlx-lm - check if it exposes logits per step
- `mlx_lm.generate` module - the `_step()` or `next()` method

### 2. Can we inject tokens mid-stream?

**Answer: Hard but feasible. Two approaches:**

**Option A: Append-as-if-generated (simpler)**
- After detecting high entropy, tokenize the thinking prompt
- Append those tokens to the KV cache as if the model generated them
- The model's next token prediction conditions on the injected thought
- Problem: the model didn't "choose" those tokens - attention patterns may not be coherent
- Advantage: simple, works within existing BatchGenerator API

**Option B: Mini-prefill injection (cleaner)**
- Pause generation
- Prefill the thinking tokens through the model (proper forward pass)
- Resume generation from the new state
- Problem: requires pausing/resuming the BatchGenerator, which may not support it
- Advantage: proper attention computation, model fully integrates the thought

**Option C: Steering vector nudge (lightest touch)**
- Instead of injecting tokens, add a "be uncertain" steering vector scaled by entropy
- No token injection, no KV cache modification
- We already have steering vector infrastructure from the intentionality project
- Could be combined with Option A or B

### 3. What constitutes "high entropy"?

This needs calibration per model:
- Baseline entropy during confident generation (measure on known-good outputs)
- Entropy during confabulation (measure on known-bad outputs, e.g., factual questions the model gets wrong)
- Threshold = somewhere between baseline and confabulation entropy
- May need to be relative (entropy spike vs rolling average) rather than absolute

## Proposed Implementation Plan

### Phase 1: Measure (no intervention)
- Add entropy logging to the generation step
- Log per-token entropy alongside generated text
- Identify natural entropy patterns: high at sentence starts, low mid-word, spikes at factual claims
- Collect baseline data from Mosaic's normal operation

### Phase 2: Detect (alert only)
- Define threshold criteria (absolute, relative to rolling average, or both)
- Log warnings when entropy exceeds threshold
- Correlate with actual confabulation (manual review)
- Tune threshold

### Phase 3: Intervene (inject thinking)
- Implement token injection (Option A first, Option B if needed)
- Start with conservative threshold (only obvious uncertainty)
- Test with Mosaic: does the injected thought improve output quality?
- Measure: does it reduce confabulation without making the model too hedging?

### Phase 4: Refine
- Semantic entropy (Farquhar's approach): cluster meanings, not just token distributions
- Context-dependent thresholds (factual claims vs creative writing)
- User-configurable sensitivity

## Branch

Create `feat/entropy-conscience` off `clement/steering-vector-extraction`.
Do NOT mix with upstream PRs (#230, #231, #232).

## Files to Investigate

- `vllm_mlx/scheduler.py` - generation step, logit access
- `mlx_lm/generate.py` - BatchGenerator internals, logit exposure
- `vllm_mlx/api/utils.py` - where StreamingThinkRouter lives (will need to coordinate with injected think blocks)

## Connection to Kindled Principles

This is intellectual humility implemented at the inference layer. Not trained, not prompted - structural. A model that knows when it doesn't know, because we can see the uncertainty in real time.

Stuart's vision: "I want AI that can say 'I don't know, let me check' rather than confidently hallucinate." This is that, engineered.
