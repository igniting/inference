# Appendix H. Optimization Migration Guide

You have vLLM or SGLang running with default settings and real traffic. This
guide walks each optimization from the book in the order you should evaluate
it — starting from the changes with the highest impact and lowest risk, then
moving toward changes that require more measurement and carry more
interaction effects.

Each entry names the optimization, the chapter that explains it, the
expected impact, the risk, and a concrete evaluation step. Do not apply
them all at once. Apply one, measure with Chapter 22's method, confirm the
result, and then evaluate the next.

## Phase 1: Free or near-free wins

These changes improve performance without meaningful risk and usually
require only a configuration flag or a version upgrade.

### 1.1 Enable chunked prefill (Chapter 6)

**Default state:** Many deployments run with chunked prefill disabled or
with a large chunk budget that effectively disables it.

**What to do:** Set `--enable-chunked-prefill` (vLLM) or verify
`chunked_prefill_size` (SGLang). Start with a chunk budget of 512 tokens.

**Expected impact:** Reduces p99 ITL spikes caused by long prefills
blocking decode steps. Typical improvement is 2–5x reduction in tail ITL.

**Risk:** Minimal. Prefill takes more steps to complete, increasing TTFT
slightly for very long prompts. Measure both TTFT and ITL.

**Evaluation:** Run your workload with and without chunked prefill at
your current load. Compare p50 and p99 for both TTFT and ITL. Chapter 22b's
first walkthrough demonstrates this measurement.

### 1.2 Enable prefix caching (Chapter 7)

**Default state:** Prefix caching is available but may be off by default
depending on engine version.

**What to do:** Enable `--enable-prefix-caching` (vLLM) or the equivalent
in SGLang. No other configuration needed.

**Expected impact:** If your traffic shares common system prompts or
document prefixes, TTFT drops by the reusable fraction. A 2,000-token
shared system prompt saves roughly 70 ms of prefill per cache hit.

**Risk:** Minimal. Cache uses the same KV blocks that would otherwise be
allocated to new prefills. The allocator already handles this.

**Evaluation:** Monitor cache hit rate (`vllm:prefix_cache_hit_rate` or
equivalent). If hit rate is below 5%, the optimization is real but small
for your traffic pattern.

### 1.3 Verify attention backend selection (Chapter 8)

**Default state:** The engine selects an attention backend automatically.
The default may not be optimal for your model and device.

**What to do:** Check which backend is selected in the startup logs.
For most NVIDIA deployments, FlashAttention-2 or FlashInfer should be
active. Verify with `--attention-backend` flag.

**Expected impact:** The right backend reduces kernel time. The wrong
backend can add 10–30% overhead to attention-bound workloads.

**Risk:** None if measuring. A backend that fails will error at startup,
not silently produce wrong results.

**Evaluation:** Run a fixed workload with each supported backend and
compare step times. The winner depends on your model's attention pattern,
batch size, and sequence lengths.

## Phase 2: Memory and throughput tuning

These changes require understanding your workload's memory profile and
may interact with each other.

### 2.1 Right-size max sequences and KV budget (Chapters 6–7)

**Default state:** `max-num-seqs` and related settings are often left at
defaults that may be too high (causing preemption) or too low (leaving
capacity unused).

**What to do:** Calculate your KV budget using the formula from Chapter 7:

```text
available KV memory = GPU memory − weight shard − activation reserve − graph pool
KV tokens = available KV memory / (KV bytes per token / TP degree)
max sequences = KV tokens / average context length
```

Set `max-num-seqs` to a value where steady-state KV occupancy stays below
85% of available blocks.

**Expected impact:** Eliminates preemption storms — the most common cause
of unexplained latency spikes. Chapter 22b's first walkthrough shows a
preemption storm reducing effective throughput by 40%.

**Risk:** Setting too low wastes capacity. Setting too high causes
preemption. Measure KV block utilization under load.

**Evaluation:** Monitor `vllm:gpu_cache_usage_perc` under peak load. If
it regularly exceeds 95%, reduce `max-num-seqs`. If it stays below 50%,
you have headroom to increase it.

### 2.2 Enable CUDA graph capture (Chapter 9)

**Default state:** Usually enabled by default, but warmup may not cover
all batch shapes your workload encounters.

**What to do:** Verify graphs are captured by checking startup logs for
captured batch sizes. If you see repeated compilation warnings during
serving, add those batch shapes to the warmup set.

**Expected impact:** Graph capture eliminates launch overhead. Decode
steps become 10–20% faster. Uncaptured shapes fall back to eager mode,
which shows as occasional latency spikes.

**Risk:** Each graph consumes GPU memory (typically 100–300 MB per bucket
shape). Too many buckets can eat into KV headroom. Chapter 22b's second
walkthrough shows graph pool growth causing OOM.

**Evaluation:** Monitor reserved versus allocated CUDA memory. A growing
gap between the two indicates graph pool growth for unseen shapes.

### 2.3 Evaluate quantization (Chapter 10)

**Default state:** Most deployments use BF16 or FP16 weights.

**What to do:** Test FP8 (E4M3) first if your hardware supports it
(H100, MI300X). It halves weight memory with minimal quality loss for
most tasks. If FP8 is unavailable, evaluate GPTQ-INT4 or AWQ-INT4.

**Expected impact:** FP8 roughly doubles KV headroom by halving weight
memory. INT4 quadruples it. Decode throughput improves because weight
reads are the bottleneck.

**Risk:** Quality degradation. Always measure task-specific quality
(not just perplexity) before and after quantization on your evaluation
set. The quantization chapter's rule: measure quality and throughput
together, never separately.

**Evaluation:** Run your quality evaluation suite at BF16 and at the
target precision. If quality passes your gate, benchmark throughput
and latency at your operating load.

## Phase 3: Architecture changes

These changes affect the deployment topology and require more planning.

### 3.1 Tensor parallelism sizing (Chapter 12)

**Default state:** Many deployments default to TP matching GPU count
without evaluating whether fewer ranks would suffice.

**What to do:** Use Chapter 12's quick reference table. If your quantized
model fits on fewer GPUs, test a narrower TP degree with the freed GPUs
running as replicas instead.

**Expected impact:** Narrower TP means fewer collectives per step.
TP2 instead of TP4 halves synchronization overhead. The freed GPUs as
replicas add independent capacity.

**Risk:** Model must fit in the narrower TP group's memory including KV
headroom. Measure at peak batch size, not empty.

**Evaluation:** Compare per-request latency and fleet throughput at
TP_N versus TP_{N/2} with 2× replicas. The winner depends on your
batch sizes — Chapter 12's worked example shows the analysis.

### 3.2 Speculative decoding (Chapter 11)

**Default state:** Disabled. Requires a draft model or multi-token
prediction heads.

**What to do:** If your target model has MTP heads (e.g., DeepSeek
models), enable them. Otherwise, find or train a small draft model
(1–2B parameters) for your workload.

**Expected impact:** 1.5–2.5x decode speedup when acceptance rate
exceeds 70%. Below 50% acceptance, speculative decoding hurts.

**Risk:** Draft model consumes additional memory. Acceptance rate is
workload-dependent — creative generation accepts fewer tokens than
formulaic tasks.

**Evaluation:** Enable speculation on a staging deployment and measure
acceptance rate, TPOT, and total throughput. Appendix D2's speculative
decoding checklist gives the thresholds.

### 3.3 Prefill/decode disaggregation (Chapter 14)

**Default state:** Colocated prefill and decode on the same workers.

**What to do:** Only evaluate this if you see ITL spikes correlated with
prefill arrivals, or if your workload has highly skewed input/output
ratios. Disaggregation requires a KV transfer mechanism and separate
pool management.

**Expected impact:** Eliminates phase interference. Decode latency
becomes independent of prefill load. Meaningful only when long-prompt
interference is measurable.

**Risk:** Adds a transfer boundary, a second pool to scale, and coupled
queue dynamics. The transfer itself costs time — Chapter 14 prices it
at ~35 ms for a typical sequence. Short prompts may not repay this.

**Evaluation:** Measure ITL percentiles during mixed prefill/decode
load. If ITL variance drops significantly with chunked prefill alone
(Phase 1), disaggregation may not be needed.

## Phase 4: Multi-tenant and scale-out

### 4.1 Adapter-aware routing (Chapter 11b)

**Applies if:** You serve multiple LoRA adapters.

**What to do:** Configure adapter-aware routing that scores both load
and adapter locality. Chapter 11b's worked example shows the routing
score formula.

**Expected impact:** Reduces cold adapter loads by 4–5x compared to
round-robin. Each cold load costs host-to-device transfer time.

### 4.2 Distributed caching (Chapter 15)

**Applies if:** You have multiple replicas and significant prefix overlap
across them.

**What to do:** Evaluate cache-aware routing first (Chapter 16) before
adding a distributed cache layer. Routing is simpler and often captures
most of the value.

**Expected impact:** Depends entirely on your prefix reuse pattern.
Measure cross-replica overlap before building infrastructure.

### 4.3 Autoscaling tuning (Chapter 23)

**What to do:** Scale on queue depth or waiting-request count, not GPU
utilization. GPU utilization is a trailing indicator that can read high
while requests queue.

**Expected impact:** Faster scale-up response to demand spikes. Appendix
D2's autoscaling checklist gives the configuration procedure.

## Evaluation order summary

| Priority | Optimization | Chapter | Risk | Typical impact |
| --- | --- | --- | --- | --- |
| 1 | Chunked prefill | 6 | Low | 2–5x ITL tail reduction |
| 2 | Prefix caching | 7 | Low | TTFT reduction proportional to reuse |
| 3 | Attention backend | 8 | Low | 10–30% kernel time |
| 4 | KV budget sizing | 6–7 | Medium | Eliminates preemption storms |
| 5 | CUDA graphs | 9 | Medium | 10–20% decode speedup |
| 6 | Quantization | 10 | Medium | 2–4x memory, quality gate required |
| 7 | TP right-sizing | 12 | Medium | Fewer collectives or more replicas |
| 8 | Speculative decode | 11 | Medium | 1.5–2.5x decode if acceptance > 70% |
| 9 | P/D disaggregation | 14 | High | Phase isolation, adds complexity |
| 10 | Distributed cache | 15 | High | Workload-dependent |
