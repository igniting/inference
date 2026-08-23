# Appendix D. Deployment Patterns and Decision Checklists

These patterns are starting points. Each diagram omits management components
such as image registries, secret stores, and deployment controllers.

## Pattern 1: One device

```text
client -> API and engine -> one accelerator
```

Use this when the model and required state fit and one device meets the SLO.
It has the fewest failure and communication paths. Add replicas before adding
model parallelism when independent capacity is the goal.

Watch CPU preprocessing, memory headroom, and the difference between cold and
steady behavior.

## Pattern 2: Replicated single-node workers

```text
                 +-> replica A (one or more local devices)
client -> router +-> replica B
                 +-> replica C
```

Use this for horizontal capacity and failure isolation. Choose routing based on
load, session affinity, adapters, and cache locality. Keep replicas in
independent failure domains where possible.

Watch fragmented caches, synchronized cold starts, and global overload.

## Pattern 3: Multi-node model-parallel replica

```text
router -> replica
          +-> node 0: ranks 0..7
          +-> node 1: ranks 8..15
```

Use this when model weights or state do not fit in one node. Map frequent tensor
groups inside fast fabrics and use pipeline or expert boundaries deliberately
across nodes.

Watch collective stragglers, pipeline bubbles, membership failure, and rank-to-
topology mapping.

## Pattern 4: Expert-parallel MoE service

```text
requests -> attention/data-parallel groups
                     |
             expert dispatch fabric
            /       |        |       \
       expert ranks and optional replicas
```

Use this when experts dominate model size and conditional compute justifies
distributed ownership. Select prefill- and decode-appropriate communication.
Collect router traces and plan expert placement or replication.

Watch hot experts, network rails, grouped-GEMM shapes, and rebalancing safety.

## Pattern 5: Prefill/decode disaggregation

```text
                 +-> prefill pool -- KV transfer --+
client -> router |                              decode pool -> stream
                 +-> colocated pool (optional) ----+
```

Use this when phase interference or phase-specific scaling limits goodput.
Retain a colocated path for requests whose transfer would not pay off if the
router can estimate the choice reliably.

Watch coupled queues, transfer failures, pool ratios, and state accumulation
between stages.

## Pattern 6: Encoder/prefill/decode

```text
media -> encoder pool -> feature transfer -> prefill pool
                                              |
                                           KV transfer
                                              |
                                              v
                                         decode pool
```

Use this for encoder-heavy multimodal traffic with independent batching or
reuse. Cache media processing and encoder outputs at the appropriate trust
boundary.

Watch feature identity, dynamic media shapes, two transfer boundaries, and
first-output attribution.

## Pattern 7: Hierarchical cache

```text
GPU cache <-> host cache <-> local storage <-> distributed cache
    ^                                             |
    +----------- router and directory ------------+
```

Use this for expensive, reusable prefixes or session state that should survive
one GPU. Separate directory metadata from bulk data transfer and make stale
locations safe.

Watch promotion traffic, write policy, cross-tenant isolation, and cache-aware
hotspots.

## Pattern 8: Post-training loop

```text
prompt source -> rollout pool -> rewards -> trainer pool
                    ^                         |
                    +---- weight transfer ----+
```

Use this when an inference engine generates online training data. Decide
whether pools are colocated, alternating, or asynchronous. Version every
trajectory and invalidate state after weight changes.

Watch long-tail groups, stale policy data, peak memory during updates, and
mixed-rank failure.

## Selecting a pattern

Begin with the simplest pattern that fits the model and SLO. Add a boundary only
when it provides measurable value through independent scaling, state reuse,
failure isolation, or hardware specialization. Every new boundary adds a queue,
a protocol, a failure mode, and an observability requirement.

## Decision checklists

This appendix collects the deployment decisions that practitioners face most
often and compresses each into a structured checklist: what inputs you need,
what procedure to follow, and where in the book the reasoning lives. Every
checklist uses Atlas as its running example (70B dense decoder, BF16, 140 GB
weights, 320 KiB KV per token, TP4 on 4x 80 GB GPUs) so the numbers are
concrete, but the procedures generalize to any model.

---

### 1. Parallelism Configuration

**Decision:** How many GPUs, and what parallelism strategy?

**Inputs:** weight size in bytes, per-GPU memory, GPUs per node, model
architecture (dense vs. MoE).

**Procedure:**

```
Does the model fit on one GPU?
  Test: weight_bytes < 0.8 x GPU_memory

  YES --> No parallelism needed.
          If tight (>0.7x), consider weight-only quantization
          to leave room for KV cache and activations.

  NO  --> Does it fit on one node with tensor parallelism?
          Test: weight_bytes / N_gpus_per_node < 0.8 x per_GPU_memory

          YES --> Use TP = N_gpus within the node.
                  Requires NVLink or equivalent between all TP ranks.

          NO  --> Two options:
                  (a) Add PP across nodes, TP within each node.
                      Budget for pipeline bubbles (~(PP-1)/micro_batches).
                  (b) Quantize to reduce weight size and retry the fit.
```

**Atlas:** 140 GB / 1 GPU = 140 > 64 (0.8 x 80). Does not fit on one GPU.
140 / 4 = 35 < 64. Fits with TP4 on one node.

**MoE models:** EP typically spans nodes (experts tolerate higher-latency
interconnect). TP stays within each node. See Ch. 14 for EP/TP interaction.

**Reference:** Ch. 13, Ch. 10, Ch. 14

---

### 2. KV Cache Sizing

**Decision:** How much memory to reserve for the KV cache, and how many
concurrent sequences you can serve.

**Inputs:** KV bytes per token (= 2 x n_layers x n_kv_heads x d_head x
dtype_bytes), max context length, target concurrent sequences, available GPU
memory after weights/activations/graphs/safety margin.

**Procedure:**

```
Step 1: Compute total KV demand.
  KV_total = KV_per_token x max_context x max_concurrent_sequences

Step 2: Compute available memory.
  available = GPU_memory - weight_shard - activation_overhead
              - graph_pools - safety_margin (5-10% of GPU memory)

Step 3: Does KV_total <= available?
  YES --> Proceed. Monitor utilization in production (checklist 10).
  NO  --> Reduce, in order of preference:
          1. max_concurrent_sequences (simplest, direct control)
          2. max_context (if workload permits)
          3. Quantize KV cache (FP8/INT8 halves KV memory)
          4. Add more GPUs
```

**Atlas:** 327,680 bytes/token x 4,096 ctx x 64 seqs = 85.9 GB. Per-GPU
available: 80 - 35 - 3 - 2 - 4 = 36 GB, x4 GPUs = 144 GB. Fits. At 128K
context the same calculation yields 2.75 TB -- must reduce concurrency,
context, or quantize KV.

**Reference:** Ch. 7

---

### 3. Chunked Prefill Configuration

**Decision:** What chunk size to use for chunked prefill.

**Inputs:** decode step time at target batch size, prefill cost per token,
ITL target from your SLO.

**Procedure:**

```
Step 1: Start with a default chunk size of 512 tokens.

Step 2: Check the latency constraint.
  decode_step_time + prefill_chunk_time <= ITL_target
  max_chunk <= (ITL_target - decode_step_time) / time_per_prefill_token

Step 3: Check prefill efficiency.
  Very small chunks waste throughput due to per-step overhead.
  Profile prefill tokens/sec vs. chunk size; the knee is typically
  around 256-1024 tokens.

Step 4: Pick the largest chunk size at or above the efficiency knee
  that still satisfies the latency constraint.
```

**Atlas:** decode ~45 ms at batch 64, prefill ~0.035 ms/token, ITL target
150 ms. Max chunk: (150 - 45) / 0.035 = 3,000. Efficiency knee at ~512.
Any value in 512-3,000 works; start at 512, increase if prefill throughput
matters more than tight ITL control.

**Caution:** As the decode batch grows, decode_step_time rises and the budget
for prefill chunks shrinks. Re-profile when you change concurrency.

**Reference:** Ch. 6

---

### 4. When to Enable Prefix Caching

**Decision:** Should you enable prefix caching (automatic prompt caching /
RadixAttention)?

**Inputs:** workload prefix patterns, memory headroom (checklist 2).

**Procedure:**

```
Is there significant prefix reuse in your workload?

  System prompts shared across requests?  --> Strong benefit.
  Multi-turn conversations?               --> Yes, benefit grows with turns.
  RAG with repeated retrieval templates?  --> Template portion is cacheable.
  Unique prompts, no shared structure?    --> Low benefit; disable to save
                                              memory and hash overhead.

If enabled, verify in production:
  - Cache hit rate (vllm:cache_hit_rate or equivalent).
    Below 10-15% means the cache is not paying for itself.
  - Watch for eviction thrashing (high churn = wrong block size
    or too little memory allocated to cache).
```

**Memory cost:** Prefix caching holds KV blocks that might otherwise be freed.
If KV cache is near capacity (checklist 2), enabling prefix caching can reduce
max concurrency. Quantized KV caches help here.

**Reference:** Ch. 7, Ch. 16

---

### 5. When to Disaggregate Prefill and Decode

**Decision:** Should prefill and decode run on separate GPU pools?

**Inputs:** ITL distribution, prefill length distribution, KV transfer
bandwidth, queue time statistics.

**Procedure:**

```
Step 1: Is prefill interfering with decode latency?
  Check: do ITL p99 spikes correlate with long-prompt arrivals?
  NO  --> Chunked prefill (checklist 3) is probably sufficient.
  YES --> Continue.

Step 2: Is the workload skewed?
  Long prompts, short outputs  --> prefill-heavy; dedicated pool helps.
  Short prompts, long outputs  --> decode-heavy; dedicated pool helps.
  Balanced                     --> less benefit; chunked prefill may suffice.

Step 3: Is the transfer cost acceptable?
  transfer_time = KV_size_for_prompt / link_bandwidth
  If transfer_time > queue_time_saved, disaggregation hurts.

Step 4: Do you have enough GPUs to staff two pools without
  creating new bottlenecks?
```

**Atlas:** A 4,096-token prompt = 1.28 GB KV. Over 25 GB/s link: 51 ms
transfer. Worthwhile if it avoids queuing behind a 2-second prefill. For short
prompts (256 tokens, 80 MB KV), the queuing delay is also short and
disaggregation adds overhead for little gain.

**Reference:** Ch. 15

---

### 6. Quantization Selection

**Decision:** Which quantization method to use.

**Inputs:** BF16 quality baseline on your task, target GPU memory budget,
hardware generation, kernel availability.

**Procedure:**

```
Step 1: Do you need to quantize?
  Model fits comfortably in BF16 with adequate KV budget --> skip.

Step 2: Try FP8 first.
  - Minimal quality loss (<0.5% on most tasks).
  - H100, MI300X, and newer. Often a single framework flag.
  - Halves weight memory vs. BF16.

Step 3: If FP8 unavailable or insufficient, try GPTQ or AWQ.
  - Weight-only INT4/INT8. Requires offline calibration.
  - Quality loss is task-dependent.

Step 4: Lower precision (INT4 weight + INT4 KV, GGUF).
  - Significant quality risk. Only for resource-constrained cases.

At every step:
  - Measure quality BEFORE and AFTER on YOUR task.
  - Verify optimized kernels exist for your hardware and shapes.
    Missing kernels cause silent fallback to slower paths.
```

**Pitfall:** A model that loses 1% on MMLU might lose 5% on your domain task.
Generic benchmarks are not sufficient; always measure on your workload.

**Reference:** Ch. 10

---

### 7. Speculative Decoding: When It Helps

**Decision:** Should you enable speculative decoding?

**Inputs:** output length distribution, available GPU memory after weights and
KV (checklists 1-2), draft model availability, grammar/constraint usage.

**Procedure:**

```
Step 1: Is the workload output-heavy?
  Short input, long output --> more opportunity for speedup.
  Long input, short output --> most time in prefill; less impact.

Step 2: Can you achieve a high acceptance rate?
  > 70%   --> likely beneficial.
  50-70%  --> marginal; profile carefully.
  < 50%   --> speculation wastes compute; disable.
  Measure on your actual workload, not generic text.

Step 3: Does the draft model fit in remaining memory?
  Its weights + KV must not evict target model KV capacity.
  If tight: use a very small draft (1-2B), Medusa heads, or EAGLE.

Step 4: Structured output or grammar constraints?
  Verify the implementation propagates grammar state to the draft
  model. Some do not, causing low acceptance on constrained output.
```

**Rule of thumb:** Helps most when memory-bound (GPU underutilized during
decode) and the draft model predicts the target well. At high batch sizes
where you are compute-bound, the extra computation may not pay off.

**Reference:** Ch. 11

---

### 8. Routing Policy Selection

**Decision:** How to distribute requests across replicas.

**Inputs:** replica count, prefix reuse pattern (checklist 4), adapter usage,
telemetry refresh interval.

**Procedure:**

```
Single replica?
  --> No routing decision. Skip.

Multiple replicas, no prefix reuse, no adapters?
  --> Least-connections (uniform request cost) or
      least-estimated-work (variable cost; estimate from
      prompt length + expected output length).

Multiple replicas with prefix reuse?
  --> Hybrid cost score balancing:
      (a) Queue depth / estimated wait at each replica.
      (b) Prefix cache hit potential at each replica.
      A cache hit saving 500 ms of prefill is worth routing to
      a slightly longer queue.

Multiple replicas with adapters (LoRA)?
  --> Add adapter-locality term. Prefer replicas with the adapter
      already loaded; otherwise prefer most idle adapter slots.

Always:
  - Add uncertainty penalty proportional to telemetry staleness.
  - Implement fallback: retry on next-best if chosen replica rejects.
```

**Reference:** Ch. 17

---

### 9. Autoscaling Configuration

**Decision:** How to configure autoscaling for inference.

**Inputs:** end-to-end startup time, traffic pattern, latency SLOs, cost
constraints.

**Procedure:**

```
Step 1: Measure startup time end-to-end.
  image_pull + weight_load + graph_compilation + warmup + health_check
  For Atlas: expect 3-8 minutes. This is your minimum reaction time.

Step 2: Choose the scale-out signal.
  DO NOT use GPU utilization -- it is a trailing indicator.
  USE: queue age, TTFT trend (rising p50/p95), or
       pending requests / available KV slots.
  Require N consecutive intervals above threshold to trigger.

Step 3: Size the warm pool.
  Must absorb spikes shorter than startup time.
  warm_pool >= peak_spike_requests / per_replica_throughput

Step 4: Configure scale-down with hysteresis.
  Scale out at queue_age > 2s.
  Scale in  at queue_age < 0.5s for 10+ minutes.
  The gap must be wide enough to prevent oscillation.

Step 5: Set minimum replica count.
  Never scale to zero unless you tolerate cold-start latency.
```

**Pitfall:** Tight scale-down hysteresis causes flapping. Each cycle wastes the
full startup time and may spike latency. When in doubt, scale down slower.

**Reference:** Ch. 17, Ch. 24

---

### 10. "Is My Deployment Healthy?" Checklist

**Decision:** Is the deployment operating within acceptable bounds?

Run after deployment, periodically, and on any alert.

```
+----+------------------------------------+-------------+------------------+
| #  | Metric                             | Threshold   | Action if bad    |
+----+------------------------------------+-------------+------------------+
| 1  | Queue age                          | < 50% of   | Scale out or     |
|    |                                    | TTFT SLO    | reduce traffic   |
+----+------------------------------------+-------------+------------------+
| 2  | KV cache utilization               | < 85%       | Reduce concurr.  |
|    |                                    |             | or add capacity  |
+----+------------------------------------+-------------+------------------+
| 3  | Preemptions (last hour)            | 0           | KV undersized;   |
|    |                                    |             | see checklist 2  |
+----+------------------------------------+-------------+------------------+
| 4  | Graph/kernel fallback rate         | 0%          | Add missing      |
|    |                                    |             | shapes to warmup |
+----+------------------------------------+-------------+------------------+
| 5  | p99 TTFT                           | < SLO       | Check prefill    |
|    |                                    |             | sched, queue,    |
|    |                                    |             | prefix caching   |
+----+------------------------------------+-------------+------------------+
| 6  | p99 ITL                            | < SLO       | Check batch size,|
|    |                                    |             | chunked prefill  |
+----+------------------------------------+-------------+------------------+
| 7  | Error rate                         | < budget    | Investigate OOM, |
|    |                                    | burn rate   | timeout, upstream|
+----+------------------------------------+-------------+------------------+
| 8  | TP rank step time variance         | < 5%        | Straggler: check |
|    | (max - min across ranks)           |             | thermal, bad GPU |
+----+------------------------------------+-------------+------------------+
```

**Key points:** Preemptions mean the scheduler evicted a running request's KV
to make room -- the evicted request recomputes from scratch, wasting GPU time.
Graph fallback means a shape was not compiled and fell back to eager mode (2-5x
slower). TP rank variance above 5% means the slowest GPU sets the pace for all
ranks via the all-reduce barrier.

**Reference:** Ch. 17, Ch. 24

---

### Decision Dependencies

Some decisions feed into others. Work through them in this order:

```
  Parallelism (1) --> KV cache sizing (2) --> Chunked prefill (3)
                            |                        |
                            v                        v
                      Prefix caching (4)      Disaggregate P/D (5)
                            |
                            v
                      Routing policy (8)

  Quantization (6)     <-- feeds back into (1) and (2) if memory is tight
  Spec. decoding (7)   <-- depends on memory headroom from (2)
  Autoscaling (9)      <-- uses latency targets affected by (3), (5), (6)
  Health check (10)    <-- validates all of the above in production
```

Start with parallelism, then KV sizing, then work through the rest. Revisit
earlier decisions when a later checklist surfaces a constraint you missed.

---

### Chapter Reference Summary

| Checklist                          | Chapters        |
|------------------------------------|-----------------|
| 1. Parallelism configuration       | Ch. 13, 10, 14  |
| 2. KV cache sizing                 | Ch. 7           |
| 3. Chunked prefill                 | Ch. 6           |
| 4. Prefix caching                  | Ch. 7, 16       |
| 5. Disaggregate prefill and decode | Ch. 15          |
| 6. Quantization selection          | Ch. 10          |
| 7. Speculative decoding            | Ch. 11          |
| 8. Routing policy                  | Ch. 17          |
| 9. Autoscaling                     | Ch. 17, 24      |
| 10. Deployment health              | Ch. 17, 24      |
