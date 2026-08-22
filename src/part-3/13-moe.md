# 13. Serving Mixture-of-Experts Models

In a dense transformer layer, every token follows the same feed-forward
network. In a mixture-of-experts layer, a router chooses a few expert networks
for each token. The model gains parameter capacity without applying every
parameter to every token: a layer with 64 experts that selects two per token
applies roughly 3 percent of its expert parameters to any given token.

For serving, the price of that conditional compute is movement and imbalance.
The router's decision turns one clean matrix multiplication into a scatter:
token representations must travel to whichever rank holds the selected expert,
and the layer cannot advance until the busiest rank finishes. Two deployments
with identical hardware and identical token counts can differ materially in
step time on routing alone. This chapter walks the dispatch path, prices the
imbalance, and treats expert placement as what it is operationally: a measured
feedback loop that moves weights to save per-step time.

## Visual map

**Each MoE layer dispatches token representations to selected experts.** The
dispatch boundary is where bytes move; everything left of it is metadata, and
everything right of it is per-expert compute whose duration you do not control
directly.

```blockdiag
flowchart LR
    T["Input tokens"] --> R["Router and top-k selection"]
    R --> D["Dispatch by expert owner"]
    D --> E1["Expert 1"]
    D --> E2["Expert 2"]
    D --> EN["Other experts"]
    E1 --> C["Combine weighted outputs"]
    E2 --> C
    EN --> C
    C --> O["Layer output"]
```

**Expert load balancing is a measured placement feedback loop.** The loop
moves weights between steps; a placement only pays for itself if the
straggler time it removes exceeds the weight movement and cache disturbance it
causes.

```blockdiag
flowchart LR
    X["Router trace"] --> L["Tokens per expert and rank"]
    L --> P["Candidate placement"]
    P --> M["Weight movement under new generation"]
    M --> V["Validate straggler and goodput change"]
    V --> X
```

| MoE quantity | Why averages mislead | Better observation |
| --- | --- | --- |
| tokens per expert | hot experts hide inside a mean | maximum and distribution per step |
| rank utilization | one rank gates layer completion | busiest-rank service time |
| dispatch bytes | topology changes path cost | bytes by source, destination, and link |
| EPLB gain | movement can exceed saved work | amortization time and goodput |

## Follow one token through an MoE layer

Assume the layer has 64 experts and selects two per token. Experts are spread
across eight GPUs. The router produces expert IDs and weights for every token
in the batch, so a batch of 8 tokens means 8 hidden vectors in, 16 expert
executions, and 16 output vectors combined back into 8.

The runtime then performs four steps:

1. group or pack token representations by selected expert. This is a
   permutation, usually built from a histogram or sort over the expert IDs:
   kernels that follow want contiguous per-expert segments, not scattered
   rows;
2. send each representation to the rank that owns the expert;
3. execute the expert networks, typically as a grouped matrix multiplication
   in which one kernel processes every expert's segment as a batched matrix
   with its own row count;
4. return and combine the expert outputs using the router weights — the
   inverse permutation followed by a weighted sum.

The two network phases are usually called **dispatch** and **combine**. Across
an expert-parallel group they resemble all-to-all communication, although
specialized implementations may use custom point-to-point patterns.

### Price one dispatch

Take hidden width 8,192 in BF16, so each routed activation is
2 bytes × 8,192 = 16 KiB. In a decode step with 64 active sequences and top-2
routing, the batch creates 128 assignments, and dispatch moves
128 × 16 KiB = 2 MiB per layer — paid again by combine, so 4 MiB per layer
per step. A dense layer of the same width moves none of this: its 64 rows
stay local and read shared weights.

Prefill multiplies the payload by sequence length. A 4,000-token prompt
creates 8,000 assignments, so dispatch alone carries 8,000 × 16 KiB = 125 MiB
per layer in one direction. Multiply by the number of MoE layers to see why
the interconnect, not the expert compute, can dominate prefill step time —
and why expert-parallel prefill wants the highest-bandwidth tier of the
fabric available.

The router's own output is tiny by comparison — two expert IDs and two
weights per token — but it decides the permutation, and the permutation
decides the message boundaries. Observability should follow the same rule the
payload does: count assignments, not unique input tokens. A trace that
records 64 tokens for the batch above underreports the movement by half.

## The slowest expert sets the pace

Routing is not perfectly balanced. A programming workload may favor different
experts from a multilingual chat workload. Even within one batch, a few experts
can receive many more tokens than others.

All ranks must finish before the layer can advance. A GPU that owns a hot expert
becomes the straggler while other GPUs wait. Average balance over an hour does
not prevent step-level imbalance: a workload can be perfectly balanced in
aggregate and still send 3× the mean to one expert every step. For streaming
requests the straggler is invisible as an average but visible as latency:
every active sequence's next token waits for the busiest rank, so step-time
variance from routing shows up directly as inter-token latency variance.

Padding each expert to a fixed capacity creates regular shapes but wastes work.
Dropping excess tokens can change model quality. Dynamic grouped GEMM avoids
some padding but must handle many small or uneven matrices.

vLLM's EPLB logging reduces rank balance to one number per step:
`balancedness = avg_tokens / max_tokens`, the ratio of the mean per-rank load
to the busiest rank's load, summed across layers. A perfectly balanced
deployment logs 1.0; the worked example at the end of this chapter logs
roughly 0.44. The number is cheap to compute and belongs in MoE dashboards
next to per-expert histograms.

### Capacity, padding, and the drop decision

Use the worked example's counts — `22, 14, 7, 6, 5, 4, 3, 3` across eight
experts, 64 assignments, a mean of 8 per expert. A common mitigation is a
fixed per-expert capacity, sized as a multiple of the mean. At 1.25× mean,
each expert gets 10 slots: 80 slots for 64 assignments, and expert 0, which
brought 22, overflows by 12. Those 12 tokens are either dropped — their
outputs for this layer become zeros or pass-throughs, which is a model-quality
decision — or the layer must handle a ragged shape anyway.

Padding to the maximum instead of dropping means sizing every expert for the
busiest one: 22 slots × 8 experts = 176 assignment slots for 64 useful ones.
About two thirds of the expert compute is padding. That is the steady-state
tax of an imbalanced placement, paid every step.

Dynamic grouped GEMM removes the padding by executing exactly the segments
that exist, at the cost of ragged shapes that vary per step — which is
precisely what complicates the CUDA-graph capture decisions of Chapter 9.
Expert-parallel load balancing attacks the problem one level up: change the
placement so the maximum falls, and neither drop nor pad.

## Expert placement is a cache problem

The simplest placement gives each expert one owner. Popular experts overload
their ranks. Replicating selected experts trades additional weight memory for
more destinations and better balance.

| Strategy | Weight memory | Balance mechanism | Choose when |
| --- | --- | --- | --- |
| single owner | baseline | none | routing is naturally flat |
| static replication | + replicas forever | spreads hot experts permanently | workload mix is stable |
| measured EPLB | + replicas, relocated over time | follows observed trace | mix drifts and steps are long |

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

### From counts to a placement

vLLM's rearrangement policy is
[`vllm/distributed/eplb/policy/default.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/distributed/eplb/policy/default.py),
adapted from DeepSeek's EPLB. Its module docstring in
[`eplb_state.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/distributed/eplb/eplb_state.py)
fixes a four-term vocabulary worth adopting verbatim: a **logical expert** is
part of the model; a **redundant expert** is an extra copy created for
balancing; a **physical expert** is any replica instantiated on a device; a
**local physical expert** is one on the current device. The docstring's
example: DeepSeek-R1 has 256 logical experts, adding 32 redundant experts
gives 288 physical experts, and 32 EP ranks hold 288 / 32 = 9 local physical
experts each.

The input is a load tensor of shape `[layers, num_logical_experts]`. The
entry point `rebalance_experts` first aggregates: in `EplbState.rearrange`,
per-rank physical loads are mapped back to logical experts with a
`scatter_add_` over `physical_to_logical_map`, summed over the observation
window, and all-reduced across ranks. Then the policy runs on the host — the
comment is explicit that "the load window and current map have to come back"
to CPU.

The policy itself is two greedy passes plus a topology split:

- `balanced_packing` sorts experts by load descending and repeatedly assigns
  each to "the lightest pack; full packs are masked out by inf" — longest
  processing time first, the classic list-scheduling heuristic.
- `replicate_experts` grows logical experts into physical slots one at a
  time, each round replicating `argmax(weight / logcnt)` — the expert
  whose load per existing replica is highest. Replication stops when physical
  slots run out.
- `rebalance_experts_hierarchical` runs three steps when the expert groups
  divide evenly across nodes: pack groups to nodes, replicate within nodes,
  then pack physical experts to GPUs, dividing each logical load by its
  replica count first ("Effective per-physical load = logical load divided by
  replica count"). When divisibility fails, the caller degenerates to global
  balancing by invoking the same function with one group and one node.

Two details matter for operations. First, `preserve_intragpu_slots`
post-processes the new mapping "so that experts that remain on the same GPU
keep their previous slot positions when possible" — an expert that did not
change ranks is not copied at all. Second, `EplbState.step` does not record
the load window continuously: `_should_record_current_step` enables recording
only when the next rearrangement (or the next logging step) is within
`expert_load_window_size` steps. The window is a ring buffer that is allowed
to hold stale entries most of the time, because only its freshest
`window_size` entries are ever read. And dummy steps still advance the
rearrangement counter — the comment explains why: "to ensure all ranks are
performing collective communication." A rank that skips a collective while
others take it deadlocks.

### Installing a placement while serving continues

A new mapping is only useful once its weights are in place, and the weight
movement must not stop serving. The two pinned implementations solve this
differently, and both are worth reading.

SGLang's
[`eplb_manager.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/eplb/eplb_manager.py)
makes rebalancing a coroutine. `on_forward_pass_end` calls `next()` on a
generator whose loop is, in effect: serve for `eplb_rebalance_num_iterations`
forward passes, then run one rebalance. The rebalance itself yields too —
layers are updated in chunks of `eplb_rebalance_layers_per_chunk`, and
`rebalance` yields between chunks, so each engine step installs one more
chunk and serving continues between installs. The mapping update in
[`expert_location.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/eplb/expert_location.py)
is correspondingly per-layer: `ExpertLocationMetadata.update` builds a layer
mask and applies `torch.where(mask_update, other_field, self_field)`, so
layers outside the current chunk keep the old mapping. Both directions of the
map — physical-to-logical and logical-to-all-physical, plus the CPU copies —
flip together for the layers in the chunk. A batch never sees a layer whose
two maps disagree.

The manager also refuses to rebalance blindly:
`_check_rebalance_needed` skips the update when the windowed average GPU
utilization exceeds `eplb_min_rebalancing_utilization_threshold`, and the
constructor asserts that the rebalance interval is at least the distribution
recorder's circular buffer size — "Otherwise, the circular buffer will
contain stale data." When a chunk's peer-to-peer transfer cannot find a rank
holding a needed replica (`p2p_missing_logical_experts` is non-empty),
`update_expert_location_with_recovery` falls back to a DRAM backup client or
a disk reload filtered through `generate_weight_name_filter` to just the
missing experts. After an elastic scale-up, the pattern changes: only rank 0
computes the new mapping — the comment explains that one owner keeps
"process-local launch topology" out of the decision — and the result is
broadcast to the expanded world, so every rank installs an identical map.

vLLM moves weights through pre-allocated buffers instead of yielding.
`rebalance_execute.py` defines `move_to_buffer` and `move_from_buffer` around
`transfer_layer`, which works one MoE layer at a time on weight tensors of
shape `(num_local_physical_experts, hidden_size_i)` — for a linear expert,
typically two tensors, up and down projection. Much of the time nothing needs
to move: `transfer_layer` returns a `TransferMetadata` carrying `is_unchanged`
and `is_received_locally` masks, and experts whose physical slot did not
change are skipped — the same no-op-copy goal as vLLM's
`preserve_intragpu_slots`, applied at transfer time. In async mode the transfer
runs on a side CUDA stream into `expert_weights_buffer`, and the main loop
only commits when `EplbState.step` sees `rebalanced` and
`_all_ranks_result_ready` both true, at which point `_move_to_workspace`
swaps the staged weights in. If the next rearrangement boundary arrives while
an async rearrangement is still in flight, `step` returns early without
resetting the counter — the rearrangement is deferred, not skipped. The
profile path is also instructive: `is_profile` performs a "dummy
rearrangement with maximum communication cost" so that `profile_run`
reserves memory for the communication buffers before real traffic arrives.

## Prefill and decode need different communication

Prefill sends many tokens through an MoE layer. Large messages can use network
bandwidth efficiently, and throughput-oriented dispatch kernels are
appropriate.

Decode may send only one token per active sequence. Messages are smaller and
latency dominates. A communication method tuned for large prefill transfers may
perform poorly here. The 2 MiB decode dispatch from the pricing section is
128 small contributions that must be assembled per rank; the 125 MiB prefill
dispatch is a bulk transfer where startup latency is noise.

Appendix A's transfer model shows how different the two regimes are on the
same fabric. With the declared NVLink-class figures (a = 20 µs,
b = 450 GB/s), the 2 MiB decode dispatch takes 20 µs + 2.1 MB / 450 GB/s ≈
25 µs — four fifths of it is the startup term. The 125 MiB prefill dispatch
takes 20 µs + 131 MB / 450 GB/s ≈ 310 µs — nine tenths of it is bandwidth.
Tuning that helps one regime barely touches the other, which is why the
low-latency and high-throughput paths exist as separate code paths rather
than one kernel with a flag.

The official [DeepEP repository](https://github.com/deepseek-ai/DeepEP) makes
this distinction explicit through expert-parallel dispatch and combine
primitives designed for high-throughput and low-latency regimes. Current vLLM
deployment documentation likewise describes separate communication choices for
prefill and decode in its
[expert-parallel guide](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/).

The placement layer has not caught up with the communication layer. vLLM's
`rearrange` carries a standing `TODO(bowen): Treat differently for prefill
and decode nodes` — at the pinned revision, one placement serves both phases
even though their traffic patterns differ. Until that changes, phase-aware
treatment lives in the communication choice, not the placement.

This is another reason phase disaggregation can help: each pool can choose the
parallel and communication plan suited to its phase.

## Overlap communication with useful compute

MoE execution contains work that can sometimes overlap. Token dispatch for one
batch can run while another batch computes experts. Shared experts — weights
applied to every token, with no router and therefore no dispatch — can execute
on a different stream from routed experts while the routed tokens are still in
flight. Combine for an earlier layer or
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
cancellation and failure: a batch that is aborted mid-dispatch leaves
partially filled receive buffers that the schedule must reclaim.

## The network topology is visible in every MoE layer

If expert ranks span nodes, each MoE layer sends token activations across the
network. Group-limited routing can encourage local destinations, but changes
the model's routing behavior and must be part of the architecture.

Replication gives the dispatcher a choice, and SGLang turns that choice into a
precomputed table. `compute_logical_to_rank_dispatch_physical_map` in
`expert_location.py` builds, for every rank, layer, and logical expert, which
physical replica that rank should prefer. `_find_nearest_expert` encodes the
preference order: if there is only one candidate, take it; otherwise prefer a
replica on the same GPU, then one on the same node — but only "when it
narrows the candidate set"; otherwise return −1. Ranks left with −1 are
assigned by `_fair_choices`, a seeded shuffle that spreads their traffic
evenly over the replicas, and an assertion verifies no −1 survived. The table
is computed once per placement generation, so topology preference costs a
lookup per assignment rather than a decision per token.

Place experts and ranks with network rails in mind. A hot expert behind one NIC
can bottleneck several GPUs. Measure dispatch and combine by source, destination,
message size, and layer. Aggregate network bandwidth can look healthy while one
rail determines step time.

## Worked example: balance the busiest rank

An eight-expert layer routes 64 tokens with counts `22, 14, 7, 6, 5, 4, 3, 3`.
With two contiguous experts per rank, rank 0 receives 36 assignments while rank 3
receives six. The four rank loads are 36, 13, 9, 6; the mean is 16, so vLLM's
balancedness statistic would log 16 / 36 ≈ 0.44 — the layer finishes at less
than half its average utilization.

Repack the pairs: `(E0, E7)` = 25, `(E2, E3)` = 13, `(E4, E5)` = 9,
`(E1, E6)` = 17. The maximum falls from 36 to 25, about a third less work on
the hot rank, while average utilization barely changes — it cannot, the total
is fixed. Only the maximum gates the layer.

What the repack costs: experts E1 and E7 swap ranks, so two experts' weights
cross the fabric. Assume each expert's weights total 1 GiB. Using the
declared NVLink-class figures from Appendix A, `transfer time = a + S/b` with
a = 20 µs and b = 450 GB/s gives 20 µs + 1.07 GB / 450 GB/s ≈ 2.4 ms per
expert, about 5 ms once for the pair. If the lighter hot rank saves even
0.25 ms per step, the movement pays for itself in 20 steps; if the workload
drifts and the saving is 0.05 ms per step, payback takes 100 steps and the
next drift may reverse the decision first. That sensitivity is why the
observation window and the rebalance interval are configuration, not
afterthoughts.

Replication is the alternative worth pricing against the repack: keep E0's
contiguous pair and copy E0 onto rank 3, splitting its 22 assignments between
the two replicas. At an even split, rank 0 falls to 25 and rank 3 rises to
17 — the same maximum as the repack, from one expert copy (≈2.4 ms) instead
of two. The trade is memory: the replica holds 1 GiB for as long as the
placement lives, and every placement generation that keeps E0 hot re-decides
whether that GiB is still earning its keep.

At hidden width 8,192 and BF16, each routed activation is 16 KiB. Top-2 routing
creates 128 assignments for 64 input tokens—roughly 2 MiB before protocol
overhead in each dispatch/combine direction. A placement that balances compute
but increases slow-link traffic can still lose: moving an expert from a
co-located rank to a cross-node replica converts local hops into network
hops for every assignment it serves.

## Practice: replay and update a placement

Replay the counts above on four ranks with two experts each. Propose a new
placement, calculate per-rank assignments and activation payload, and predict
the straggler. Compute the balancedness ratio for both placements and check it
moves the way the maximum moves. Then design a generation-safe EPLB update
that cannot mix old and new mappings within a batch, and decide which layers
you would move first if the update had to be chunked.

Compare prefill and decode traces and include weight-movement cost. The worked
placement is in [Appendix G](../appendices/g-worked-solutions.md#13-expert-trace-and-placement).

Expert serving makes phase differences especially pronounced. Chapter 14
generalizes the idea of assigning different stages to different worker pools.
