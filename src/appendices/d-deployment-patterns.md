# Appendix D. Deployment Patterns

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
