# 13. Serving Mixture-of-Experts Models

In a dense transformer layer, every token follows the same feed-forward
network. In a mixture-of-experts layer, a router chooses a few expert networks
for each token. The model gains parameter capacity without applying every
parameter to every token.

For serving, the price of that conditional compute is movement and imbalance.

## Follow one token through an MoE layer

Assume the layer has 64 experts and selects two per token. Experts are spread
across eight GPUs. The router produces expert IDs and weights for every token in
the batch.

The runtime then performs four steps:

1. group or pack token representations by selected expert;
2. send each representation to the rank that owns the expert;
3. execute the expert networks, often with grouped matrix multiplication;
4. return and combine the expert outputs using the router weights.

The two network phases are usually called **dispatch** and **combine**. Across
an expert-parallel group they resemble all-to-all communication, although
specialized implementations may use custom point-to-point patterns.

## The slowest expert sets the pace

Routing is not perfectly balanced. A programming workload may favor different
experts from a multilingual chat workload. Even within one batch, a few experts
can receive many more tokens than others.

All ranks must finish before the layer can advance. A GPU that owns a hot expert
becomes the straggler while other GPUs wait. Average balance over an hour does
not prevent step-level imbalance.

Padding each expert to a fixed capacity creates regular shapes but wastes work.
Dropping excess tokens can change model quality. Dynamic grouped GEMM avoids
some padding but must handle many small or uneven matrices.

This is why expert utilization, tokens per expert, and per-rank completion time
belong in MoE observability.

## Expert placement is a cache problem

The simplest placement gives each expert one owner. Popular experts overload
their ranks. Replicating selected experts trades additional weight memory for
more destinations and better balance.

A placement controller can collect routing statistics and periodically move or
replicate experts. This is expert-parallel load balancing, often abbreviated
EPLB. Changes must be coordinated: routers need the new location map, weights
must be available before traffic moves, and in-flight batches must finish under
a consistent mapping.

Reacting too quickly to a noisy batch can create movement churn. Reacting too
slowly leaves hotspots. Use a stable observation window and include the cost of
reconfiguration.

At the pinned revisions, vLLM implements EPLB state, policy, communication, and
rebalance execution under
[`vllm/distributed/eplb`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/distributed/eplb).
SGLang's corresponding manager, algorithms, distribution tracking, and location
updates live under
[`sglang/srt/eplb`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/eplb).

## Prefill and decode need different communication

Prefill sends many tokens through an MoE layer. Large messages can use network
bandwidth efficiently, and throughput-oriented dispatch kernels are
appropriate.

Decode may send only one token per active sequence. Messages are smaller and
latency dominates. A communication method tuned for large prefill transfers may
perform poorly here.

The official [DeepEP repository](https://github.com/deepseek-ai/DeepEP) makes
this distinction explicit through expert-parallel dispatch and combine
primitives designed for high-throughput and low-latency regimes. Current vLLM
deployment documentation likewise describes separate communication choices for
prefill and decode in its
[expert-parallel guide](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/).

This is another reason phase disaggregation can help: each pool can choose the
parallel and communication plan suited to its phase.

## Overlap communication with useful compute

MoE execution contains work that can sometimes overlap. Token dispatch for one
batch can run while another batch computes experts. Shared experts can execute
on a different stream from routed experts. Combine for an earlier layer or
batch can overlap later work when dependencies allow it.

Systems use names such as two-batch overlap, dual-batch overlap, or single-batch
overlap for different schedules. The name matters less than the timeline.

Draw the operations and dependencies:

```text
batch A: route -> dispatch -> expert compute -> combine
batch B:          route -> dispatch -> expert compute -> combine
```

Overlap improves utilization only if the tasks use compatible resources. A
communication kernel that consumes many streaming multiprocessors can compete
with expert GEMM. Extra in-flight batches also need more buffers and complicate
cancellation and failure.

## The network topology is visible in every MoE layer

If expert ranks span nodes, each MoE layer sends token activations across the
network. Group-limited routing can encourage local destinations, but changes
the model's routing behavior and must be part of the architecture.

Place experts and ranks with network rails in mind. A hot expert behind one NIC
can bottleneck several GPUs. Measure dispatch and combine by source, destination,
message size, and layer. Aggregate network bandwidth can look healthy while one
rail determines step time.

## Build an expert trace

Collect router outputs for representative prompts without storing user content.
For each layer and step, record token counts per expert and rank. Replay the
trace through candidate placements and estimate dispatch bytes, busiest-rank
work, and redundant-expert memory.

Then compare the estimate with engine timelines. Test prefill and decode
separately. Include the cost and safety of an EPLB update. The goal is not
perfect balance; it is lower goodput cost after communication and movement are
included.

Expert serving makes phase differences especially pronounced. Chapter 14
generalizes the idea of assigning different stages to different worker pools.
