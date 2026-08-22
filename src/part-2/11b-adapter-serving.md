# 11b. Adapter Serving and Multi-Tenant Customization

A single base model can serve many customers, but not all customers want the
same model. A legal team needs answers tuned for contract language. A medical
group needs clinical tone and terminology. A retailer needs product-catalog
fluency. Full fine-tuning would give each customer a private 140 GB checkpoint,
and the fleet would need one replica per customer. Low-rank adapters offer a
different deal: each customer gets a small weight delta -- a few hundred
megabytes -- that modifies the shared base model at inference time. One replica
serves many customers from one copy of the base weights.

The economics are attractive enough that production deployments routinely serve
dozens to hundreds of adapters on a shared fleet. The engineering is less
simple than the pitch. Adapter weights compete with KV state for the same HBM.
Scheduling must decide which adapters share a batch. The router must know which
replicas hold which adapters. Prefix sharing, speculative decoding, CUDA-graph
capture, and quantization all interact with the adapter dimension in ways that
are invisible until they produce wrong answers or surprise latency. This
chapter treats adapter serving as the scheduling, memory, and routing problem
it becomes at scale.

## Visual map

**Adapter weights are thin overlays on the shared base model.**

```blockdiag
flowchart LR
    B["Base weights (read-only, 140 GB)"] --> F["Forward pass"]
    A1["Adapter A weights (~160 MiB)"] --> F
    A2["Adapter B weights (~160 MiB)"] --> F
    F --> O["Per-request output"]
```

**Multi-adapter scheduling groups requests by active adapter within a mixed
batch.**

```blockdiag
flowchart TB
    Q["Waiting requests with adapter tags"] --> S["Adapter-aware scheduler"]
    S --> G1["Group: Adapter A requests"]
    S --> G2["Group: Adapter B requests"]
    S --> G3["Group: base-only requests"]
    G1 --> M["Mixed batch with per-request adapter pointers"]
    G2 --> M
    G3 --> M
    M --> E["Execute one step"]
```

**Adapter placement turns routing into a three-term cost.**

```blockdiag
flowchart TB
    R["Request with adapter tag"] --> C["Candidate replica"]
    C --> QT["Estimate queue time"]
    C --> PT["Estimate missing-prefix compute"]
    C --> AT["Estimate adapter-load time"]
    QT --> SC["Combined routing score"]
    PT --> SC
    AT --> SC
    SC --> D["Choose destination"]
```

| Adapter concern | Interacts with | Observable cost |
| --- | --- | --- |
| weight memory | KV cache budget | fewer concurrent sequences |
| batch composition | scheduler, CUDA graphs | switching or padding overhead |
| cache identity | prefix sharing | invalid reuse across adapters |
| cold loading | routing, TTFT | hundreds of ms on first use |
| graph capture | compilation warm-up | multiplicative graph count |

## What an adapter adds to a forward pass

A LoRA adapter decomposes a weight update into two low-rank matrices. Where the
base model applies a weight matrix W of shape `[d, d]`, the adapted model
computes `W*x + B*A*x`, where A is `[r, d]` and B is `[d, r]` for rank r.
Typical serving ranks are 8 to 64; rank 16 on Atlas's hidden size of 8,192
gives A and B matrices of `8192 x 16` each. At BF16, one such pair costs
`2 x 8192 x 16 x 2 = 512 KiB`. Applied to attention projections (Q, K, V, O)
across 80 layers, a rank-16 adapter totals about `4 x 512 KiB x 80 = 160 MiB`.
Extending to MLP layers (gate, up, down projections) roughly doubles that to
320 MiB -- still less than 0.25 percent of the 140 GB base model.

The compute is proportionally small. Each adapted layer adds two matrix
multiplications of rank r against the batch. At rank 16 the extra FLOPs per
token per layer are `2 x 2 x 8192 x 16 = 524,288` -- about 0.4 percent of
the base layer's `2 x 8192 x 8192 = 134 million` FLOPs for a single
projection. The adapter's arithmetic is noise against the base model's work.
The systems cost is not in compute; it is in memory, identity, and placement.

## Memory accounting for a multi-adapter fleet

### Base weights are shared; adapter weights add up

The 140 GB base model is loaded once per replica and never modified during
serving. Every adapter request reads the same base weights -- without this
sharing, serving N customers would require N copies. But each adapter is an
additional allocation:

```text
50 adapters x 160 MiB (rank 16, attention only) = 8 GB
50 adapters x 320 MiB (rank 16, attention + MLP)  = 16 GB
```

Sixteen gigabytes is not fatal on an 80 GB device, but it is not free either.
Chapter 7 showed that Atlas's KV budget is roughly 35 GiB after base-model
weights, activation buffers, and graph pools. Sixteen gigabytes of adapter
weights cuts that budget nearly in half, reducing maximum concurrency from
about 57 resident 8,000-token conversations to about 30. The adapter memory
competes with KV cache for the same HBM, and the competition has a clear loser:
every gigabyte spent on adapter weights is a gigabyte that cannot hold KV
state, which means fewer concurrent sequences, more preemption, or both.

### Tiered storage: hot, warm, cold

Not all adapters are equally active. Traffic typically follows a power law: a
handful of popular adapters serve most requests while a long tail sees
occasional use. A tiering strategy matches storage cost to access frequency:

| Tier | Location | Capacity | Load latency |
| --- | --- | --- | --- |
| hot | GPU HBM | limited by KV competition | 0 (already resident) |
| warm | host CPU memory | tens to hundreds of GB | tens of ms (PCIe transfer) |
| cold | disk or network | effectively unlimited | hundreds of ms to seconds |

A rank-16 adapter at 160 MiB transfers over PCIe Gen5 x16 (64 GB/s) in about
2.5 ms. In practice, the transfer is not the whole cost: the engine must
allocate destination buffers, update pointer tables, and potentially
invalidate CUDA graphs. Measured cold-load times for typical adapters run 20
to 100 ms from host memory -- the number Chapter 16 priced at 800 ms includes
disk-resident adapters on a cold path with no pipelining.

### S-LoRA's unified paging

The [S-LoRA paper](https://arxiv.org/abs/2311.03285) observed that adapter
pages and KV pages share the same management problem: both are variable-size
allocations that arrive and depart with requests, both benefit from paging to
avoid fragmentation, and both compete for the same physical memory. S-LoRA's
Unified Paging allocates adapter weight pages from the same pool as KV pages,
using the same block table machinery Chapter 7 described. The Punica kernel
executes batched LoRA operations where each sequence in the batch may use a
different adapter, gathering the correct A and B matrices through indirection
rather than requiring all sequences to share one adapter.

This unification has a scheduling consequence. The block allocator now manages
two kinds of tenants -- KV blocks and adapter blocks -- and admission must
account for both. A request that arrives for a cold adapter needs adapter
blocks allocated before its first prefill token, and those blocks reduce the
KV capacity available to every other request in the batch.

### vLLM's pre-allocated LoRA buffers

vLLM takes a different approach at the pinned revision. The `--max-loras`
flag sets the maximum number of adapters active in one batch, and
`--max-lora-rank` sets the maximum rank. The engine pre-allocates fixed
buffers sized to hold that many adapters at that rank, reserving the memory
at startup rather than paging it dynamically. The trade-off is fragmentation
for predictability: the buffers are always allocated whether or not they are
full, but the engine never needs to page adapter weights during a step.

At the pinned commit, vLLM's LoRA implementation lives under
[`vllm/lora`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/lora).
The `LoRAManager` coordinates adapter loading, and the
[`punica_wrapper`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/lora/punica_wrapper)
contains the batched LoRA kernels that apply different adapters to different
sequences in a single matrix multiplication.

SGLang's adapter support at the pinned revision lives under
[`srt/lora`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/lora).

## Scheduling with adapter awareness

### Batching across adapters

The simplest approach batches all requests together regardless of adapter.
The base-model forward pass runs once for the full batch, and each sequence's
adapter contribution is added through gathered low-rank products. This is what
the Punica kernel enables: the base matmul is one operation; the adapter
matmuls are a second, indexed operation that reads different A and B matrices
per sequence. The batch sees one base-weight read and many small adapter
reads.

The cost model is straightforward. The base-weight read dominates at 140 GB
per step; adapter weights add at most a few hundred megabytes of extra reads.
Mixed-adapter batching is therefore nearly free in steady state, provided all
adapters are already resident.

### Switching cost: I/O, not compute

The cost appears when an adapter is not resident. Loading a warm adapter from
host memory takes tens of milliseconds. Loading a cold adapter from disk
takes hundreds. During that time, either the request waits (adding to TTFT)
or the engine stalls (adding to every request's ITL).

The switching cost is I/O-bound, not compute-bound. A rank-16 adapter's
160 MiB is a memory transfer, not a matrix multiplication. This means it can
overlap with compute: while the GPU executes early layers, the engine can
transfer a cold adapter's weights for later layers. Layer 40's adapter
weights are not needed until the forward pass reaches layer 40, so they can
arrive while layers 0 through 39 execute. This is the same overlap principle
Chapter 14 applied to KV transfer in disaggregated serving, and the
opportunity is better here because adapter weights are smaller than
multi-gigabyte KV images.

### Adapter-aware grouping

When adapter-loading cost is non-trivial, the scheduler can reduce it by
grouping requests that share an adapter. If ten requests for adapter A and
two for adapter B are waiting, scheduling all ten A-requests together avoids
loading adapter B until the next step. This is adapter-affinity scheduling:
prefer to fill the batch with requests that share already-resident adapters.

The risk is starvation. If adapter A is popular and adapter B is rare,
strict affinity scheduling can delay B-requests indefinitely. Fair
scheduling across adapters requires the same discipline Chapter 6 applied
to priority classes: bound the maximum wait time, reserve minimum batch
slots for underserved adapters, or use weighted round-robin across adapter
groups. The popularity power law makes this concrete: if 80 percent of
requests use the top 5 adapters, the remaining 45 adapters share 20 percent
of batch capacity and need protection from indefinite deferral.

## Routing and adapter placement

### Each replica holds a different adapter set

In a fleet of replicas, not every replica needs every adapter. If the hot set
is 5 adapters covering 80 percent of traffic, those 5 should be resident on
every replica. The remaining 45 can be distributed: some replicas hold
adapters 6 through 25, others hold 26 through 50. A request for adapter 37
routes to a replica that already has it, avoiding the cold-load penalty.

This is the adapter term in Chapter 16's routing score:

```text
cost(R) = queue(R) + missing_tokens(R) x 0.06 ms
        + adapter_load(R) + risk(R)
```

The `adapter_load(R)` term is zero when the target replica already holds the
requested adapter and nonzero -- potentially hundreds of milliseconds --
when it does not. Chapter 16 priced the loss: sending a request to a replica
without its adapter costs 800 ms of foreground load time on first use, which
dominates both the queue and missing-prefix terms in most scenarios.

### Power-law traffic and placement strategy

Adapter popularity follows a power law. Zipf with exponent near 1.0 is a
reasonable model: the most popular adapter sees roughly 50 times the traffic
of the median adapter. The placement strategy follows:

- **Universal hot set.** The top few adapters are resident everywhere. Their
  per-replica memory cost is small (5 adapters at 160 MiB = 800 MiB), and
  their traffic share justifies the HBM.

- **Partitioned warm set.** The next tier is distributed across replicas.
  Each replica holds a subset, and the router sends requests to replicas that
  have the right adapter. The partition should be rebalanced as popularity
  shifts.

- **On-demand cold tail.** Rarely used adapters stay on disk or in a shared
  store. A request for a cold adapter pays the full load penalty, but it
  happens infrequently enough that the fleet-level impact is small.

### Why naive round-robin fails

Round-robin routing ignores the adapter dimension entirely. With 50 adapters
and 8 replicas, every adapter eventually receives traffic on every replica.
Each replica must eventually load all 50 adapters, spending
`50 x 160 MiB = 8 GB` of HBM on adapter weights -- regardless of whether
most of those adapters see only one request per hour on that replica. Worse,
each cold-load event adds tens to hundreds of milliseconds to the affected
request's TTFT, and the cold loads are scattered unpredictably across the
fleet.

Adapter-aware routing concentrates each adapter's traffic on a small number
of replicas, keeping the per-replica footprint proportional to actual use.
The router needs one additional piece of telemetry: each replica's adapter
inventory.

## Interactions with other engine mechanisms

### KV cache identity

Chapter 7 established that cached KV state depends on the full identity:
model version, tokenizer, token IDs, positions, and adapter. Two requests
with identical prompts but different adapters produce different KV state,
because the adapter modifies the attention projections that generated the
keys and values. Sharing cached prefix blocks across adapters is invalid --
the same text under a different adapter is a different model, and reuse
produces silently wrong outputs.

This means prefix sharing in a multi-adapter deployment is scoped to
requests that share both the same prompt prefix and the same adapter. The
cache hit rate drops as the adapter count grows: with 50 adapters, a system
prompt cached under adapter A benefits only the fraction of traffic using
adapter A. The total cache value of a prefix is its per-adapter hit rate
times the number of adapters that share it -- and for adapter-specific
prefixes, that number is one.

### Speculative decoding

A draft model proposes tokens that the target model verifies. When the target
model uses an adapter, the draft must produce proposals consistent with the
adapted model's distribution. A base-model draft proposing tokens for an
adapted target will have systematically lower acceptance rates wherever the
adapter has shifted the distribution, reducing the speedup or eliminating it
entirely.

The options are: apply the same adapter to the draft model (doubling the
adapter's memory footprint), use a draft model fine-tuned alongside the
adapter (requiring one draft per adapter, which rarely exists), or accept
lower acceptance and let the adaptive controller from Chapter 11 reduce or
disable speculation for adapter traffic. The third is simplest in practice.

### Quantization

Adapter weights may use a different numerical format from the base model.
The base model might be quantized to INT4 for memory savings while the
adapter weights remain in BF16 for quality. The kernel must handle
mixed-precision arithmetic: dequantize the base weights, add the BF16
adapter contribution, and accumulate in a wide type. This is the same
concern Chapter 10 raised for mixed quantization generally, but adapters
make it per-request: one batch may mix INT4-base-plus-BF16-adapter sequences
with INT4-base-only sequences.

### CUDA graphs

Chapter 9 showed that CUDA-graph capture keys include adapter state. At the
pinned vLLM revision, graph keys are the cross product of batch sizes and
LoRA counts: `product(cudagraph_capture_sizes, lora_cases)`. With 5 batch
sizes and 4 LoRA cases, the engine captures 20 graphs. Each capture takes
about 0.7 seconds, so warm-up is 14 seconds -- tolerable for startup, but
the graph pool grows with the product.

Adapter switching can invalidate a captured graph. A graph captured with
adapter A's weight pointers baked in will silently apply adapter A's weights
to every sequence, regardless of which adapter the sequence actually uses.
Dynamic pointer resolution -- reading adapter addresses from a buffer rather
than baking them into the graph -- avoids this but requires the graph to
include the indirection. Chapter 9's observation applies directly: a captured
graph with baked adapter weights silently serves the wrong model.

## Worked example: Atlas adds fifty customer adapters

### Price the memory

Atlas base model: 140 GB in BF16. Each rank-16 adapter across the four
attention projections (Q, K, V, O) in 80 layers:

```text
Per adapter:
  4 projections x 80 layers x 2 matrices x 8192 x 16 x 2 bytes
  = 4 x 80 x 2 x 8192 x 16 x 2
  = 4 x 80 x 512 KiB
  = 160 MiB

50 adapters:
  50 x 160 MiB = 8 GB
```

Total weight footprint: 140 + 8 = 148 GB. On a 4-way tensor-parallel
deployment across 80 GB devices, each device holds 35 GB of base weights and
2 GB of adapter weights, leaving about 43 GB for KV cache, activations, and
graph pools. Compared to the adapter-free 45 GB, the adapter overhead is
about 4.5 percent of device memory -- modest for 50 customers.

If adapters also cover MLP projections (gate, up, down), each adapter doubles
to about 320 MiB, and 50 adapters cost 16 GB total or 4 GB per device.
The KV budget drops from 45 to 41 GB per device, a loss of roughly 5
additional 8,000-token conversations per device.

### Adapter-aware routing saves cold-load time

Assume Zipf-distributed traffic with exponent 1.0 across 50 adapters, served
by 8 replicas.

**Naive round-robin.** Each replica receives traffic for all 50 adapters.
Assuming each adapter's first arrival on a replica costs 50 ms to load from
host memory, the fleet pays `8 replicas x 50 adapters x 50 ms = 20,000 ms`
of cumulative cold-load time during warm-up. More importantly, 400
individual requests (one per adapter per replica) each suffer an extra 50 ms
added to their TTFT -- potentially breaching the 600 ms target on a
request that was otherwise on budget.

**Adapter-aware routing.** Partition the 50 adapters across replicas: the
top 5 adapters (covering roughly 45 percent of traffic under Zipf-1.0) are
resident everywhere. The remaining 45 are distributed in groups of about 6
per replica. Each replica holds 5 + 6 = 11 adapters, totaling
`11 x 160 MiB = 1.76 GB` per replica instead of 8 GB. Cold-load events
drop from 400 to `8 x 11 = 88` during warm-up, and the router avoids
cold loads in steady state by directing each request to a replica that
already holds its adapter.

The steady-state benefit compounds. Under round-robin, adapter 50 (the
least popular) might arrive at each replica once per hour, and if eviction
has reclaimed its slot, every arrival pays a cold load. Under adapter-aware
routing, adapter 50 lives on one replica, sees all its traffic there, and
stays warm as long as it receives any traffic at all.

### How the routing score changes

Take a concrete request for adapter 37, which is in the cold tail:

| Replica | Queue | Missing prefix | Adapter load | Total |
| --- | ---: | ---: | ---: | ---: |
| R0 (has adapter 37) | 200 ms | 0 ms | 0 ms | 200 ms |
| R1 (idle, no adapter 37) | 0 ms | 0 ms | 50 ms | 50 ms |
| R2 (light load, no adapter 37) | 80 ms | 0 ms | 50 ms | 130 ms |

Without the adapter term, R1 wins at 0 ms. With it, R1 costs 50 ms and
R2 costs 130 ms. R0 at 200 ms loses either way in this snapshot --
but if R0's queue clears before the adapter load on R1 completes, R0
delivers the first token sooner. The adapter term changes the winner and
prevents the fleet from scattering cold loads across replicas that will
never see a second request for that adapter.

## Practice: simulate adapter-aware vs. adapter-blind routing

Generate a synthetic trace: 50 adapters with Zipf-distributed popularity
(exponent 1.0), 8 replicas, 1,000 requests arriving at Poisson intervals.
Each adapter is 160 MiB. Cold load from host memory costs 50 ms. Each
replica starts with no adapters loaded.

Simulate two routing strategies:

1. **Round-robin** (adapter-blind): requests cycle through replicas in
   order, ignoring adapter state.

2. **Adapter-aware**: the top 5 adapters are pre-loaded on all replicas.
   Remaining adapters are assigned to replicas by hashing the adapter ID.
   The router sends each request to a replica in the adapter's assigned
   set, breaking ties by shortest queue.

Measure:

- Total cold-load events across the fleet.
- Number of requests with TTFT exceeding 600 ms (assuming base TTFT is
  400 ms, so a 50 ms cold load is safe but a second stacked load is not).
- Per-replica adapter memory high-water mark.
- 99th-percentile TTFT for each strategy.

The worked calculation is in
[Appendix G](../appendices/g-worked-solutions.md#11b-adapter-routing-simulation).

Adapter serving is a memory, scheduling, and routing problem with a
distinctive shape: the weight overhead per adapter is small, but the
interaction with every other mechanism in Part II is not. Adapters change
what prefix sharing means, what CUDA-graph keys contain, what the routing
score must include, and how much HBM the KV cache actually gets. The next
part extends these single-engine concerns across multiple accelerators and
machines, where adapter placement becomes a distributed-scheduling problem.
