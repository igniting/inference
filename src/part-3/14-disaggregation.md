# 14. Disaggregated Serving

A colocated LLM worker performs both prefill and decode. That arrangement keeps
state local and makes one worker responsible for the whole request. It also
forces two different workloads to share the same queue, hardware, and parallel
configuration — and the two workloads disagree on all three. Prefill wants
large batches and compute efficiency; decode wants steady, small steps and
memory bandwidth. Sharing a queue means one of them is always mis-served.

Disaggregated serving separates stages so each can be scheduled and scaled
independently. The common case places prefill on one worker pool and decode on
another. The price is a new stage in the middle — KV state must move between
pools — and a new coupling: neither pool can be sized correctly without
measuring the other.

## Visual map

**Prefill/decode separation turns one engine queue into a stage pipeline.**
The feedback edge on the right is what keeps the pipeline honest: admission
must know whether decode capacity actually exists, not whether prefill is idle.

```blockdiag
flowchart LR
    A["Admission"] --> P["Prefill queue and workers"]
    P --> K["KV state transfer"]
    K --> D["Decode queue and workers"]
    D --> O["Output stream"]
    D --> B["Decode capacity signal"]
    B --> A
```

**Conditional placement chooses between local reuse and a transfer boundary.**

```blockdiag
flowchart TB
    R["Request shape and queue state"] --> X{"Split saves more interference than transfer costs?"}
    X -->|No| C["Colocated prefill and decode"]
    X -->|Yes| P["Remote prefill"]
    P --> T["Versioned KV transfer"]
    T --> D["Reserved decode slot"]
```

**One transfer, four commitments.** The destination reserves before data
moves, the source publishes into the reservation, both sides poll for
completion, and only a verified transfer inserts the request into a decode
batch. Every earlier state is reversible; insertion is not.

```blockdiag
flowchart LR
    R["Request admitted to decode"] --> A["Reserve destination blocks"]
    A --> PB["Publish block map to prefill"]
    PB --> T["Prefill transfers into reserved blocks"]
    T --> V{"Poll: Success on all ranks?"}
    V -->|Yes| I["Insert into decode batch"]
    V -->|No or Failed| X["Release reservation; retry or re-prefill"]
```

| Stage | Capacity variable | New failure mode | Admission signal |
| --- | --- | --- | --- |
| Prefill | prompt tokens per second | output state piles up | estimated service time |
| Transfer | bytes and concurrent copies | partial or timed-out KV | reserved bandwidth and deadline |
| Decode | active sequences and context | no slot after prefill | predicted decode availability |
| Output | client consumption | backpressure retains state | buffer age and disconnect |

## Why split prefill from decode?

Prefill benefits from large compute-efficient operations. Decode values steady,
low-latency steps and sufficient memory bandwidth. A long prompt running beside
active decoders can create an output stall. A tensor-parallel plan that helps
prefill may add too much synchronization to decode.

The stall is not a corner case — it falls out of the service-time model
directly. Using the worked example's `prefill_ms = 20 + 0.035 × tokens`, one
6,000-token prompt occupies its worker for roughly 230 ms. Every decode step
that would have run during those 230 ms is late, so every sequence sharing
that worker sees an inter-token gap larger than the entire prefill. Against
an ITL budget of 150 ms, a single long prompt on a colocated worker is by
itself an SLO breach for all its neighbors. Chunked scheduling softens the
spike by interleaving prefill and decode steps, but the bytes still compete
for the same compute and the same memory bandwidth.

With separate pools, a prefill worker processes the prompt and produces the
initial KV state. That state moves to a decode worker, which generates the rest
of the response.

```text
request -> prefill queue -> prefill workers
                              |
                           KV transfer
                              |
                              v
          stream <- decode workers <- decode queue
```

Separation also makes each phase's parallel plan a free choice. Prefill can
use wider tensor parallelism to shorten the 230 ms; decode can stay at a
narrow plan where synchronization overhead dominates. Chapter 12 priced the
difference: two collectives per layer means 160 collective launches per step
at any tensor width above one, costing ~3.2 ms of startup alone when decode
batches are small — overhead decode pays every 45 ms step, while prefill's
large payloads amortize it. Colocated deployment forces one compromise on
both phases.

The [DistServe](https://arxiv.org/abs/2401.09670) and
[Splitwise](https://arxiv.org/abs/2311.18677) papers study phase splitting as a
way to reduce interference and select phase-specific resource plans.

## The transfer is part of latency

The KV cache for a long prompt can be large. Disaggregation helps only if the
state can be transferred, registered, and made visible before the saved
interference or better placement pays for that cost.

A transfer protocol needs several pieces of metadata: request identity, model
and cache format, source and destination ranks, block ranges, memory addresses
or handles, and completion state. The sender and receiver must agree on how
tensor-parallel or context-parallel shards map between them. Real deployments
add wrinkles the metadata must survive: prefill and decode pools running
different attention-TP sizes, pipeline stages that own only a layer range, and
cache layouts that are not flat layer-indexed lists.

Bulk data should move on a path designed for it, while control messages arrange
the rendezvous. A push design lets the prefill side initiate. A pull design lets
decode request the blocks. Both need timeouts, cancellation, and idempotent
cleanup.

If transfer fails, the system can retry, choose another decode worker, recompute
prefill, or fail the request. The policy should depend on the remaining deadline
and expected recompute cost. The costs are computable. Retrying the transfer
costs another ~95 ms of link time plus whatever queue delay applies;
recomputing the prefill costs the full ~230 ms again plus the freed worker's
opportunity cost; failing the request spends everything already paid. Against
the 600 ms TTFT budget, a request that has already waited 200 ms in queues
cannot afford recomputation and barely affords one retry — which is why the
policy belongs to admission, which knows the elapsed budget, not to the
transfer layer, which does not.

### Chunked sends hide the boundary

Nothing requires the full 1.83 GiB to exist before transfer starts. SGLang's
sender interface exposes the seam directly: `should_send_kv_chunk(num_pages,
last_chunk)` decides per step whether ready pages go out now, and the
prefill scheduler's `send_kv_chunk` path streams completed blocks while later
chunks are still computing. The default is eager — "return num_pages > 0" —
and each chunk's readiness is tracked separately until a final
`last_chunk=True` send concludes the request.

The overlap is worth real milliseconds. Suppose chunks complete every 2,000
tokens. The first chunk is ready at 20 + 0.035 × 2,000 = 90 ms and its
~610 MiB take 12 + 610 MiB / 22 GiB/s ≈ 39 ms on the link — finished at
~129 ms, while prefill still has 140 ms of compute left. The last chunk
leaves at ~230 ms and needs only its own slice of link time, so the
pipeline's transfer tail shrinks from the serial 95 ms to roughly
12 ms plus the final chunk's bandwidth time, about 40 ms total. End-to-end,
chunked transfer lands near 270 ms instead of 325 ms — without changing a
single byte moved. The trade is protocol state: partial transfers hold
reservations longer, and every retirement path must clean up pending chunk
bookkeeping or leak it.

### Five states from handshake to insert

SGLang's disaggregation stack is built from five small abstract roles in
[`base/conn.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/disaggregation/base/conn.py):
a `BaseKVManager` holding transfer state, a `BaseKVSender`, a
`BaseKVReceiver`, and a `BaseKVBootstrapServer` that lets pools find each
other. The transfer's entire lifecycle is one poll enum:

```text
KVPoll: Failed=0, Bootstrapping=1, WaitingForInput=2, Transferring=3, Success=4
```

The scheduler polls senders and receivers every step; nothing blocks. The
`KVArgs` the manager carries are deliberately raw — `kv_data_ptrs`,
`kv_data_lens`, `kv_item_lens`, layer ids, an `ib_device` — because the
transports move GPU memory by address, not by copying through intermediate
buffers. The comments catalog the layout hazards metadata must express:
per-tensor TP slice dims "used when prefill/decode attn_tp_size differ",
`prefill_start_layer`/`prefill_end_layer` for pipeline-parallel sub-ranges,
and auxiliary state types (`MAMBA`, `SWA`, `swa_ring`) beyond plain KV.

Two details reward attention. First, the handshake is
reserve-then-publish: the decode side's `DecodePreallocQueue` allocates
destination blocks before data moves and tracks
"`_num_published_destinations` — destinations visible to prefill but not yet
on the transfer queue." Prefill never sends into blocks decode has not
committed. Admission into the handshake is itself queued: requests wait in a
`PrefillBootstrapQueue` until the bootstrap server resolves pool endpoints,
and `_check_if_req_exceed_kv_capacity` rejects up front any request whose
KV indices cannot fit the destination pool — failing fast instead of
mid-transfer. Second, completion is a distributed agreement, not a local fact:
`process_disagg_prefill_inflight_queue` all-reduces the poll states across
the attention TP/CP groups, and in pipeline parallelism a later rank treats a
non-terminal poll as "undone" when an earlier rank already saw
Success/Failed — the comment attributes the mismatch to "clock skew or
propagation delay" and chooses to wait rather than crash.

The failure path is equally explicit. A `Failed` poll routes to
`handle_inflight_transfer_failure`, which releases the request's KV
reservation, aborts the request with an internal-server-error status,
increments a `transfer_failed_reqs` counter, and streams the error to the
client. A separate optimistic path — `optimistic_release_and_requeue`,
gated by `should_force_retry` — releases the destination and re-queues the
request for another prefill attempt instead of failing it. And one
bookkeeping rule prevents a slow leak: `clear_pending_chunk_send` must run on
every path that retires a request without a final chunk, because "a stale
entry holds the unified-memory compaction gate closed for the process
lifetime."

Observability is part of the interface: `get_transfer_metric` returns
`KVTransferMetric` with `transfer_latency_s`, `alloc_latency_s`, and
`transfer_total_bytes` — and the docstring admits "backends that cannot
isolate transfer latency can leave this as None." A dashboard that averages
over Nones silently lies; the schema makes the gap visible instead. The poll
loop itself is honest about its rough edges — one inline comment reads
"todo: set Transferring correctly in backend," so callers treat
`WaitingForInput` and `Transferring` alike as "still in flight." And on the
decode side, insertion is FIFO through `pop_preallocated`, which refuses to
run at all under pipeline parallelism unless the caller supplies consensus
rids — the error message says it plainly: "PP consensus is required when
pp_size > 1."

## Two queues create a coupled system

Separating stages does not remove queueing. It creates a queue before prefill, a
transfer boundary, and a queue before decode.

If prefill produces requests faster than decode can consume them, completed KV
state accumulates while users wait for a decode slot. Scaling prefill harder
would make the system worse. If decode is overprovisioned, expensive workers
wait for prefills.

The correct pool ratio depends on arrival rate, prompt and output lengths,
cache hits, and each phase's service time. Measure the whole pipeline. A low
prefill queue can hide a growing decode queue, and the symptoms are specific:
completed-but-uninserted KV state holding reservations, rising
transfer-queue age, decode admission lagging prefill completion. Each is the
signature of prefill outproducing decode admission, visible one stage before
users feel it.

Raw throughput may fall after adding transfer while goodput rises because TTFT
and ITL become more predictable. State the metric. “Disaggregation increases
throughput” and “disaggregation never increases throughput” are both too broad.

### Sizing the pools

Appendix A's `Q = λW` makes the coupling concrete. Assume the worked
example's 6,000-token prompts, decode steps of 45 ms, batches of 32, and
400-token outputs. One prefill worker completes a request every ~230 ms, so
its production rate is λ_p ≈ 4.3 requests per second per worker. A decode
worker retires its whole batch only when sequences finish: with 400-step
generations, each sequence occupies a slot for 400 × 45 ms = 18 s, so a
32-slot decode worker turns over 32 / 18 s ≈ 1.8 requests per second.
Steady state needs prefill production to match decode turnover: one prefill
worker feeds roughly two decode workers at this mix. Every parameter moves
the ratio — longer outputs raise decode residency, cache hits raise
effective prefill capacity, and a burst of short prompts can flip the
bottleneck within minutes.

The transfer stage sits between them with its own queue and its own capacity:
at 12 ms setup plus 22 GiB/s, concurrent transfers share a pipe that a
6,000-token prompt occupies for ~95 ms. Ten concurrent such transfers
serialize into a second of link time. Admission should treat transfer
bandwidth the way Chapter 5 treats GPU occupancy — a schedulable resource
with its own queue, not a free side effect of finishing prefill.

## Conditional and dynamic disaggregation

Not every request should cross a stage boundary. A short prompt may be cheaper
to run entirely on a colocated worker. A request whose prefix is already cached
on a decode worker may skip remote prefill. A large prompt with a tight ITL SLO
may benefit most from separation.

A conditional router compares local execution with remote prefill plus transfer
and queueing. The worked example gives the comparison shape: remote costs
~230 ms prefill + ~95 ms transfer + a decode-queue wait; colocated costs
~230 ms prefill on a worker whose decode neighbors each absorb an
SLO-breaching stall. The transfer is worth it when the stall it removes —
spread across the sequences that would have shared the worker — exceeds the
95 ms and the added queueing. Short prompts flip the comparison: at 500
tokens, prefill is ~38 ms and the KV state only ~156 MiB, but the transfer
setup alone is 12 ms, nearly a third of the phase it enables.

A dynamic system can also change pool membership as workload
phase ratios change. The comparison the router runs, per request class:

| Request class | Colocated cost | Remote cost | Usually wins |
| --- | --- | --- | --- |
| short prompt, no cache hit | small prefill, no transfer | setup-dominated transfer | colocated |
| short prompt, cached on decode worker | full prefill again | prefix reuse, no or tiny send | remote (cache) |
| long prompt, tight ITL neighbors | stall breaches neighbor ITL | 95 ms boundary + queueing | remote |
| long prompt, idle cluster | 230 ms uncontended | same + transfer tail | colocated |

The last row is the honest one: under low load, disaggregation adds latency
and removes nothing, because there is no interference to remove. Conditional
placement is not a per-request optimization only — it is how the deployment
stays correct across load levels. Reconfiguration must account for warm weights, graph
capture, cache loss, and draining — the same invalidation cascade as
Chapter 12's mesh changes, because a worker changing pools changes its
parallel plan.

## Encoder, prefill, and decode

Multimodal models add an encoder stage. Large video or vision encoders can
dominate first-output latency and use different hardware shapes from language
decode. Separating encoder, prefill, and decode produces an E/P/D topology.

Encoder outputs must now move to prefill, followed by KV state moving to decode.
The two movements differ in kind: encoder output is a fixed-size embedding set
per image — the same size whether the caption is five words or five hundred —
while prefill KV scales linearly with text. Different elasticity is exactly why
one shared pool serves both poorly: the encoder pool sizes for media throughput,
the prefill pool for token throughput, and neither's queue says anything useful
about the other.
The additional boundary is worthwhile only when independent batching, caching,
or hardware assignment pays for it. Repeated questions about the same media can
make an encoder cache especially valuable: the encoder output for an
unchanged image is deterministic, so a hit removes the dominant first-stage
latency without touching the language pipeline.

The idea generalizes beyond language. A diffusion pipeline can place its text
encoder, denoiser, and decoder in different pools. A rollout system can separate
generation from training and weight distribution. Disaggregation is stage
placement, not a feature unique to KV caches.

## What the implementations reveal

vLLM defines connector interfaces and several transfer or offload integrations
under
[`distributed/kv_transfer`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/distributed/kv_transfer).
SGLang's prefill, decode, staging, and transport implementations live under
[`srt/disaggregation`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/disaggregation).

Both code trees contain several backends and compatibility checks. That is a
useful warning: a transport name alone does not guarantee support for a model's
cache layout, parallel sizes, speculative mode, or device. vLLM's connector
factory registers the integrations by name, and the names themselves teach
the design space: `NixlConnector` alongside separate `NixlPushConnector` and
`NixlPullConnector` variants — this chapter's push-versus-pull choice made
literal — plus `MultiConnector` for composing several tiers and
`OffloadingConnector` for moving KV to host memory rather than another GPU
pool.

The [Mooncake paper](https://arxiv.org/abs/2407.00079) extends the design around
a KV-centric architecture using GPU, CPU, memory, and storage resources. It
also highlights early rejection under overload, connecting disaggregation back
to goodput and admission control.

### One interface, two sides

vLLM's
[`kv_connector/v1/base.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
organizes the same lifecycle as one class with two halves, and its module
docstring is the cleanest statement of the boundary. Scheduler-side
primitives decide what should happen: `get_num_new_matched_tokens` reports
how many tokens exist in a remote cache — with the explicit contract that it
"might be called multiple times for a given request and should be
side-effect free," because the scheduler probes speculatively;
`update_state_after_alloc` reacts to buffer allocation;
`request_finished` decides who owns the blocks now — it "returns whether KV
cache should be freed now or if the connector now assumes responsibility for
freeing the blocks asynchronously," which is how a cache tier can outlive
its request; `take_events` exports the same KV events Chapter 15's
distributed cache consumes.

Worker-side primitives do the moving, at layer granularity:
`start_load_kv` begins async loads, `wait_for_layer_load` blocks until layer
i has arrived — so transfer overlaps the forward pass instead of preceding
it — and `save_kv_layer`/`wait_for_save` mirror the pattern on the way out.
`handle_preemptions` lets the connector react when the scheduler retracts
requests whose blocks are mid-transfer, and `get_finished` reports which
async sends and receives have completed so the scheduler can act on them.
Capability is declared, not discovered: `requires_kv_delivery` states
"whether this connector hands off KV that must be reliably delivered," and a
`SupportsHMA` marker class flags connectors that handle hybrid memory
attention — the machine-checkable form of the compatibility warning above.
The split enforces a rule this chapter has been circling: scheduling
decisions live in one process, bytes move in another, and metadata is the
only thing that crosses.

## Worked example: price the state boundary

Use `prefill_ms = 20 + 0.035 × tokens`, a 45 ms decode step, and a KV link with
12 ms setup plus payload at 22 GiB/s. The 6,000-token Atlas prompt creates about
1.83 GiB of KV state — 6,000 × 320 KiB = 1,875,000 KiB. Ideal transfer is
therefore roughly 95 ms, compared with
about 230 ms of prefill.

Walk the placement decision for one such request. Remote: 230 ms prefill
(assuming a free prefill worker) + 95 ms transfer + one decode step of 45 ms
before the first token — about 370 ms of pipeline time before output starts,
against a 600 ms TTFT budget that leaves roughly 230 ms for both queues.
Chunked sends at 2,000-token granularity pull that to roughly 315 ms — the
transfer overlaps prefill instead of following it — and hand back 55 ms of
queue budget without moving one byte less.
Colocated on an otherwise idle worker: 230 ms and no transfer — but "idle"
is the assumption that fails under load. The same request placed beside
eight decoding sequences inflicts a 230 ms stall on all of them; eight
sequences × one breached 150 ms ITL budget is the interference cost the
transfer is buying down. Disaggregation wins here not because 95 ms is
cheap but because the colocated alternative is worse for everyone sharing
the worker.

Now give the transfer a failure. If the link drops at the moment prefill
finishes, the request has ~370 ms of pipeline time committed and roughly 230 ms
of TTFT budget left. A retry costs another 95 ms plus queueing — feasible if
the decode reservation survived, impossible if it was released. Recomputation
costs 230 ms of prefill again plus re-transfer — over budget. The right
policy at this point is a bounded retry with reservation held; the wrong time
to decide that is during the failure. Deadlines belong in the admission
record so the failure handler can compare costs instead of guessing.

That boundary is material. Disaggregation wins only if isolating decode from
prefill interference and improving pool utilization repays the transfer and a
new queue. An idle prefill worker is not capacity when no decode slot will be
available afterward.

## Practice: simulate three placements

Build prefill, transfer, and decode service-time tables from the functions
above. Compare colocated, always-disaggregated, and conditional placement for
short and 6,000-token prompts under bursts.

Report every stage queue, TTFT, ITL, goodput, transferred bytes, failure cleanup,
and idle capacity. Derive a conditional split threshold. The worked model is in
[Appendix G](../appendices/g-worked-solutions.md#14-prefilldecode-split).

Disaggregation creates a new question: if valuable state can move between
workers, should it survive beyond the request? Chapter 15 builds a distributed
cache around that question.
