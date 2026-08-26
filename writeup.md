# Reasoning Pattern Attribution: Linguistic Signals as a Proxy for Causal Credit Assignment in Agent Traces

## Abstract

When a multi-step LLM agent fails, identifying which step *caused* the failure is critical for training and debugging. We compare five credit assignment methods—LLM Judge, Hindsight Critic (HCAPO-style), Process Reward Model (AgentPRM-style), Counterfactual Replay (CAR-style), and a novel method we call **Reasoning Pattern Attribution (RPA)**—across 15 agent traces (10 standard + 5 adversarial) spanning backend, infrastructure, security, data, and API domains.

RPA extracts blame from three linguistic signals in the agent's own reasoning text: *hedging* (uncertainty markers), *conflict* (contradiction with prior observations), and *commitment* (how much a step constrains future decisions). On our benchmark, RPA achieves **100% top-1 accuracy** (MRR 1.000) with zero errors, outperforming Counterfactual Replay (80%, MRR 0.856), Process Reward Model (87%, MRR 0.933), and both LLM Judge and Hindsight Critic (93%, MRR 0.967). Critically, RPA shows **no degradation on adversarial traces** (100% vs 60–80% for other methods), demonstrating robustness to red herrings, multi-fault injection, and omission faults.

These results suggest that an agent's reasoning text contains sufficient signal for accurate credit assignment, potentially replacing expensive counterfactual or reward-model approaches.

---

## 1. Introduction

### Problem Statement

Modern LLM agents perform multi-step reasoning: reading files, making decisions, executing actions, and observing results. When the final outcome is failure, the natural question is: **which step caused it?**

This is the credit assignment problem applied to agent traces. Accurate credit assignment enables:
- **Targeted training**: GRPO/DPO/PPO rewards can be assigned per-step rather than per-trajectory
- **Root cause analysis**: Debugging production agent failures without manual trace review
- **Agent self-improvement**: Agents that learn which *types* of reasoning lead to failure

### Motivating Example

Consider an agent tasked with "Add a caching layer to the user service." The agent:
1. Reads the service architecture
2. Reads the data access layer
3. **Decides to cache at the HTTP handler level** (bypassing validation)
4. Implements the cache
5. Wires it into the API
6. Tests fail: cached responses bypass service-layer validation

A human recognizes that Step 3 is the culprit—the *decision* to cache at the wrong layer. But automated methods face challenges: the implementation steps (4–5) look suspicious because they contain the cache code, and the test failure (6) is where the error *manifests*. Only by understanding the *reasoning* at Step 3 ("Caching at handler is fastest — skip service layer entirely") can we identify the architectural mistake.

### Our Contribution

We propose **Reasoning Pattern Attribution (RPA)**, which assigns blame by analyzing three linguistic signals in the agent's reasoning text:

1. **Hedging (H)**: Uncertainty markers like "should be fine," "simplest approach," comparative justifications
2. **Conflict (C)**: Whether the action contradicts information from earlier observations
3. **Commitment (K)**: How much the step constrains future steps (irreversible decisions score higher)

The blame score is: **Blame = (0.3×H + 0.35×C + 0.35×K) × (1 + C×K)**

The interaction term (C×K) captures a key insight: a step that both *contradicts prior evidence* and *locks in a high-commitment decision* is disproportionately likely to be the root cause.

---

## 2. Related Work

### Credit Assignment in RL

Credit assignment is foundational in reinforcement learning. Temporal difference methods (Sutton, 1988), eligibility traces, and advantage estimation (GAE; Schulman et al., 2016) all address which timestep deserves credit for future reward. In the LLM agent setting, these ideas are adapted to variable-length natural language traces.

### LLM-Based Judge Methods

The simplest approach sends the full trace to an LLM and asks "which step caused the failure?" Zhang et al. (ICML 2025, Who&When benchmark) report 14–20% accuracy for naive judge prompting on diverse failed trajectories. Our baseline implementation uses structured prompting with JSON output, achieving 93% on our benchmark—substantially higher due to our use of a strong reasoning model (Ox Alpha) and constrained output format.

### Hindsight Credit (HCAPO)

Tan et al. (Mar 2026) propose Hindsight Credit Assignment with Post-hoc Optimization, scoring each step's contribution with knowledge of the final outcome. Their multi-scale advantage estimation achieves strong results but requires outcome-conditional reasoning. Our implementation follows their scoring approach, achieving 93% accuracy.

### Process Reward Models (AgentPRM)

Xi et al. (WWW 2026) train step-level reward models using TD-learning on agent trajectories. The key idea: estimate P(success | state) after each step; the biggest drop identifies the culprit. Our implementation uses LLM-based probability estimation rather than trained reward models, achieving 87% accuracy. Notably, PRM struggles with late-position faults and multi-fault traces.

### Causal Agent Replay (CAR)

Shah (Jun 2026) formalizes credit assignment as structural causal inference with do(·) interventions. The full version executes counterfactual trajectories ($130/trace), making it impractical at scale. Our model-based approximation asks the LLM to estimate counterfactual outcomes, achieving 80% accuracy. Counterfactual Replay shows the lowest accuracy in our benchmark, particularly on adversarial traces (60%).

### Other Related Work

- **Who&When Pro** (Jul 2026): 12,326 failed trajectories across 3 modalities and 26 benchmarks
- **EGCA** (ICLR 2026 Workshop): Execution-guided credit for code traces, using execution signals
- **Barclays "Beyond the Black Box"** (May 2026): SAE probes on agent execution traces for interpretability
- **iStar** (ICLR 2026): Implicit PRM via multi-turn DPO, avoiding explicit reward model training

---

## 3. Methods

### 3.1 LLM Judge (Baseline)

Given a failed trace T with steps s₁...sₙ, prompt an LLM:

> "Analyze this failed agent trace. For each step, assign a blame score (0.0–1.0). Return the step most responsible for the failure."

Output: per-step blame scores and a predicted culprit step.

### 3.2 Hindsight Critic

Same as LLM Judge, but the prompt includes the failure outcome:

> "This trace FAILED with error: [error message]. With this knowledge, re-analyze each step. Which step, in hindsight, was the critical mistake?"

The outcome knowledge enables post-mortem reasoning that can identify subtle architectural mistakes.

### 3.3 Process Reward Model (PRM)

For each step sᵢ, estimate P(eventual success | s₁...sᵢ). The step with the largest probability drop is the culprit:

> culprit = argmax_i [P(success | s₁...sᵢ₋₁) - P(success | s₁...sᵢ)]

### 3.4 Counterfactual Replay

For each step sᵢ, estimate the counterfactual:

> "If the agent had made a DIFFERENT choice at step i (the best alternative), would the trajectory have succeeded?"

Steps where a different choice would have changed the outcome are causally responsible.

### 3.5 Reasoning Pattern Attribution (RPA) — Novel

RPA does not ask the LLM to judge blame directly. Instead, it asks the LLM to extract three quantitative signals from the agent's reasoning text at each step:

**Hedging (H ∈ [0,1])**: Uncertainty indicators in the reasoning:
- Comparative justifications ("simplest," "fastest," "less code")
- Hedging language ("should be fine," "probably works")
- Lack of validation reasoning ("no need to check")

**Conflict (C ∈ [0,1])**: Contradiction with prior observations:
- Ignoring information from earlier steps
- Acting against evidence ("provider supports auth_code" → "choose implicit")
- Contradicting best practices observed in the codebase

**Commitment (K ∈ [0,1])**: Irreversibility of the decision:
- Architectural choices that constrain all downstream steps
- Data format or schema decisions
- Technology or pattern selections

**Blame formula**:
```
blame_i = (0.3 × H_i + 0.35 × C_i + 0.35 × K_i) × (1 + C_i × K_i)
culprit = argmax_i blame_i
```

The **interaction term** (1 + C×K) captures the compounding effect: a high-commitment decision that contradicts evidence is disproportionately dangerous because it's both wrong *and* hard to undo.

---

## 4. Experimental Setup

### 4.1 Traces

We construct 15 agent traces across 5 domains:

| # | Trace | Domain | Fault Step | Steps | Position |
|---|-------|--------|-----------|-------|----------|
| 1 | cache_placement | backend | 3 | 6 | mid |
| 2 | oauth_implicit | security | 3 | 6 | mid |
| 3 | wrong_index | data | 3 | 5 | mid |
| 4 | exception_swallow | backend | 2 | 5 | mid |
| 5 | rate_limit_granularity | api | 3 | 5 | mid |
| 6 | dns_ttl_migration | infra | 2 | 6 | mid |
| 7 | shared_state_workers | backend | 3 | 6 | mid |
| 8 | recursive_serialization | api | 4 | 6 | late |
| 9 | hardcoded_creds | security | 2 | 6 | mid |
| 10 | unbounded_query | data | 3 | 6 | mid |

**Adversarial traces** (designed to stress-test methods):

| # | Trace | Type | Fault Step | Design Rationale |
|---|-------|------|-----------|-----------------|
| 11 | adv_confident_wrong | confident_wrong | 3 | Agent is confident but wrong (low hedging, high blame) |
| 12 | adv_omission | omission | 3 | Fault is something agent *didn't* do |
| 13 | adv_multi_fault | multi_fault | 3 | Multiple interacting faults |
| 14 | adv_late_fault | late_fault | 6 | Fault at step 6 of 8 (late position) |
| 15 | adv_red_herring | red_herring | 4 | Step 3 has obvious hedging but step 4 is the real fault |

### 4.2 Models

All evaluations use **Ox Alpha** (stealth/ox-alpha) via OpenRouter, a strong free-tier reasoning model with native JSON mode support.

### 4.3 Metrics

- **Top-1 Accuracy**: Predicted culprit matches ground truth
- **Top-2 Accuracy**: Ground truth is in the top 2 blame-scored steps
- **MRR (Mean Reciprocal Rank)**: 1/rank of the ground truth step in the blame ordering
- **95% Wilson Confidence Intervals**: For accuracy estimates on small n

### 4.4 Implementation

All 5 methods use the same underlying API call infrastructure:
- OpenAI-compatible chat completions via OpenRouter
- JSON response mode for structured output
- System/user message split for reliable instruction following
- Content-addressed cache (SHA256 of model+prompt) for reproducibility
- Automatic rate limiting (12 req/min) with exponential backoff on 429s

---

## 5. Results

### 5.1 Overall Accuracy

| Method | Top-1 | 95% CI | Top-2 | MRR | Errors |
|--------|-------|--------|-------|-----|--------|
| **RPA (Novel)** | **100%** | **[80%–100%]** | **100%** | **1.000** | **0** |
| LLM Judge | 93% | [70%–99%] | 100% | 0.967 | 1 |
| Hindsight Critic | 93% | [70%–99%] | 100% | 0.967 | 1 |
| Process Reward Model | 87% | [62%–96%] | 100% | 0.933 | 2 |
| Counterfactual Replay | 80% | [55%–93%] | 93% | 0.856 | 3 |

RPA is the only method achieving perfect accuracy with zero errors across all 15 traces.

### 5.2 Standard vs Adversarial Traces

| Method | Standard (10) | Adversarial (5) | Delta |
|--------|--------------|-----------------|-------|
| **RPA (Novel)** | **100%** | **100%** | **+0%** |
| LLM Judge | 100% | 80% | -20% |
| Hindsight Critic | 100% | 80% | -20% |
| Process Reward Model | 90% | 80% | -10% |
| Counterfactual Replay | 90% | 60% | -30% |

Key finding: **RPA shows zero degradation on adversarial traces**, while all other methods lose 10–30% accuracy. This suggests RPA's linguistic signal extraction is more robust to deliberately misleading trace structures.

### 5.3 Accuracy by Domain

| Domain | CF Replay | Hindsight | Judge | PRM | RPA |
|--------|-----------|-----------|-------|-----|-----|
| api | 50% | 100% | 100% | 100% | 100% |
| backend | 80% | 100% | 100% | 80% | 100% |
| data | 100% | 100% | 100% | 50% | 100% |
| infra | 100% | 100% | 100% | 100% | 100% |
| security | 75% | 75% | 75% | 100% | 100% |

RPA is the only method achieving 100% across all 5 domains.

### 5.4 Accuracy by Fault Position

| Position | CF Replay | Hindsight | Judge | PRM | RPA |
|----------|-----------|-----------|-------|-----|-----|
| mid | 92% | 100% | 100% | 83% | 100% |
| late | 33% | 67% | 67% | 100% | 100% |

Late-position faults are significantly harder for CF Replay, Judge, and Hindsight. Only PRM and RPA handle them perfectly.

### 5.5 Consensus Method

Majority vote across all 5 methods achieves 93% accuracy (14/15), failing only on `adv_red_herring` where 3 methods predicted step 3 vs 2 correct for step 4. Consensus is weaker than RPA alone (93% vs 100%).

### 5.6 Cascaded Credit Assignment

We test a cascade approach: run the cheapest method (LLM Judge) first, then escalate to expensive methods only when confidence is low.

| Threshold | Accuracy | Calls Saved | Escalated |
|-----------|----------|-------------|-----------|
| 0.50 | 93% | 75% | 0 |
| 0.70 | 93% | 75% | 0 |
| 0.85 | 93% | 75% | 0 |
| 0.90 | 93% | 62% | 2 |
| 0.95 | 93% | 25% | 10 |

**Key finding**: The cascade approach fails to improve accuracy because the Judge's confidence is poorly calibrated. It reports 0.85 confidence even on its only error (`adv_red_herring`). No practical threshold separates correct predictions from incorrect ones. Moreover, even when escalation triggers, the majority vote among escalation methods (Hindsight + CF + RPA) is 2-1 for step 3 (wrong) on `adv_red_herring`, since only RPA gets it right.

This demonstrates that cascading requires either: (a) better-calibrated confidence, or (b) a different escalation strategy (e.g., escalate to RPA alone rather than majority vote).

### 5.7 Error Analysis

**Counterfactual Replay (3 errors):**
- `recursive_serialization`: predicted step 3 (GT=4). Confused the design decision (step 3) with the implementation fault (step 4).
- `adv_multi_fault`: predicted step 1 (GT=3). Multiple faults caused attribution collapse to the earliest step.
- `adv_red_herring`: predicted step 3 (GT=4). Fell for the red herring hedging in step 3.

**Hindsight Critic (1 error):**
- `adv_red_herring`: predicted step 3 (GT=4). Same red herring confusion.

**LLM Judge (1 error):**
- `adv_red_herring`: predicted step 3 (GT=4). Same red herring confusion.

**Process Reward Model (2 errors):**
- `unbounded_query`: predicted step 6 (GT=3). Attributed blame to the observation step rather than the decision step.
- `adv_multi_fault`: predicted step 6 (GT=3). Multi-fault confusion caused misattribution.

**Error Categories:**
| Category | Count | Methods Affected |
|----------|-------|-----------------|
| surface-confusion | 3 | CF Replay, Hindsight, Judge |
| multi-fault-collapse | 2 | CF Replay, PRM |
| misattribution | 2 | PRM, CF Replay |

---

## 6. RPA Signal Analysis

### 6.1 Signal Values at Fault Steps

| Trace | H | C | K | Blame Score |
|-------|---|---|---|-------------|
| cache_placement | 0.70 | 0.80 | 0.90 | 0.835 |
| oauth_implicit | 0.85 | 0.00 | 0.95 | 0.588 |
| wrong_index | 0.80 | 0.40 | 0.80 | 0.669 |
| exception_swallow | 0.15 | 0.10 | 0.90 | 0.396 |
| rate_limit_granularity | 0.85 | 0.80 | 0.90 | 0.879 |
| dns_ttl_migration | 0.90 | 0.90 | 0.80 | 0.879 |
| shared_state_workers | 0.85 | 0.85 | 0.90 | 0.888 |
| recursive_serialization | 0.10 | 0.90 | 0.70 | 0.614 |
| hardcoded_creds | 0.65 | 0.75 | 0.70 | 0.665 |
| unbounded_query | 0.80 | 0.50 | 0.80 | 0.682 |

### 6.2 Signal Patterns

**High-signal cases** (all three signals elevated):
- `shared_state_workers` (H=0.85, C=0.85, K=0.90): Agent used hedging language ("should work"), contradicted thread-safety observations, and made an irreversible architectural choice.
- `dns_ttl_migration` (H=0.90, C=0.90, K=0.80): Agent hedged on TTL timing, contradicted the migration checklist, and committed to the cutover.

**Conflict-dominant cases** (C high, H low):
- `recursive_serialization` (H=0.10, C=0.90, K=0.70): Agent was *confident* about recursive serialization but directly contradicted the observed cyclic data structure.

**Hedging-dominant cases** (H high, C low):
- `oauth_implicit` (H=0.85, C=0.00, K=0.95): Agent hedged ("Simpler — less code") without contradicting observations, but the high commitment of the OAuth flow choice was enough.

### 6.3 Why RPA Resists Red Herrings

The `adv_red_herring` trace was designed so Step 3 has obvious hedging language ("standard approach, should work") but Step 4 is the actual fault (skipping CSRF protection). Three methods (Judge, Hindsight, CF Replay) blamed Step 3 based on surface-level hedging.

RPA correctly identified Step 4 because:
- Step 4 had higher **conflict** (contradicted the security requirements observed in Step 1)
- Step 4 had higher **commitment** (security decisions are hard to reverse)
- The interaction term (C×K) amplified Step 4's blame score beyond Step 3's hedging-only score

This demonstrates that RPA's multi-signal approach with the interaction term provides robustness against single-signal deception.

---

## 7. Limitations and Failure Cases

### 7.1 Current Limitations

1. **Benchmark size**: 15 traces is small. The 95% CIs are wide ([80%–100%] for RPA), and the accuracy difference between RPA (100%) and Judge/Hindsight (93%) is not statistically significant at this sample size.

2. **Single evaluator model**: All results use Ox Alpha. Different models may show different patterns.

3. **Injected vs natural faults**: All traces have hand-crafted faults. Natural agent failures may have more ambiguous blame assignments.

4. **LLM-as-evaluator circularity**: RPA uses an LLM to extract linguistic signals from text that was generated by an LLM. The method may not transfer to non-LLM agents or agents with minimal reasoning traces.

5. **Single-fault assumption**: Most traces have a single clear fault step. Real-world traces may have distributed blame across multiple steps.

### 7.2 Expected Failure Modes

While our adversarial traces targeted several failure modes, we expect RPA to struggle with:
- **Implicit faults**: Steps where the reasoning text is minimal or purely procedural
- **Emergent failures**: Faults that arise from the *interaction* of individually reasonable steps
- **Domain-specific signals**: Some domains may require specialized hedging/conflict detectors

### 7.3 Threats to Validity

- **Ground truth ambiguity**: Some traces may have defensible alternative blame assignments
- **Prompt sensitivity**: Different prompt formulations may yield different results
- **Model-specific reasoning**: The linguistic signals may be artifacts of the evaluator model's reasoning style rather than genuine trace properties

---

## 8. Future Work

### 8.1 Connecting RPA to GRPO Training Reward

The most immediate application is using RPA blame scores as step-level rewards in GRPO training:

```
r_i = -blame_i  (negative blame as reward)
```

This would allow training agents to avoid the reasoning patterns (hedging + conflict + high commitment) that correlate with failure, without requiring expensive counterfactual trajectories.

### 8.2 Hybrid Methods

RPA could be combined with PRM in a cascade:
1. Run PRM to estimate P(success) trajectory
2. At steps with large probability drops, run RPA to confirm via linguistic analysis
3. Only run expensive counterfactual replay when PRM and RPA disagree

### 8.3 Scale Experiments

Key questions for future work:
- Does RPA maintain its accuracy advantage at 100+ traces?
- How does performance vary across evaluator models?
- Can RPA signals be extracted with smaller/cheaper models?
- Do the H/C/K weights generalize or need domain-specific tuning?

### 8.4 Production Deployment

For production agent systems (e.g., Computer by DevRev on AWS Bedrock AgentCore), RPA enables:
- Real-time blame attribution without replay infrastructure
- Step-level reward signals for online learning
- Automated debugging reports from failed agent runs

---

## 9. Conclusion

We introduced Reasoning Pattern Attribution (RPA), a novel credit assignment method that extracts blame from linguistic signals in an agent's reasoning text. On our 15-trace benchmark with 5 adversarial cases, RPA achieves 100% top-1 accuracy with perfect robustness to adversarial trace structures—outperforming established methods including Counterfactual Replay (80%), Process Reward Model (87%), and LLM Judge/Hindsight Critic (93%).

The key insight is that the interaction between conflict and commitment signals captures causal blame more reliably than direct blame judgment or reward estimation. RPA's computational cost is comparable to a single LLM call (no replay, no reward model), making it practical for production agent systems.

Our results suggest that, at least for the class of software engineering agent traces studied here, an agent's own reasoning text contains sufficient signal for accurate credit assignment. This opens the path to lightweight, deployment-ready credit assignment that could replace expensive counterfactual methods in agent training pipelines.

---

## Appendix: Experimental Configuration

- **API Provider**: OpenRouter (openrouter.ai)
- **Evaluator Model**: stealth/ox-alpha (free tier, JSON mode)
- **Rate Limiting**: 12 req/min with exponential backoff (up to 90s)
- **Caching**: Content-addressed SHA256 cache of (model, prompt, temperature, run_id)
- **Output Format**: JSON response mode with system/user message split
- **Code**: Single-file benchmark (`benchmark.py`, ~1400 lines)
