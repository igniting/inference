# Appendix E. Glossary

Definitions in this glossary describe concepts as used in the book. Frameworks
may use the same term differently.

**Acceptance rate** — The fraction of speculative proposal tokens accepted by
the target model. It does not include draft or verification cost and is not a
speedup by itself.

**Admission control** — The decision to accept, delay, redirect, or reject new
work based on capacity and service objectives.

**All-gather** — A collective in which every rank receives the shards held by
all ranks.

**All-reduce** — A collective that reduces values across ranks and returns the
result to every rank.

**All-to-all** — A collective in which each rank sends a distinct portion of
data to every other rank. Expert dispatch and combine often use this pattern.

**Arithmetic intensity** — Operations performed per byte moved across a chosen
memory boundary.

**Attention backend** — A selected implementation of attention for a device,
model pattern, dtype, cache layout, and execution mode.

**Batch invariance** — A numerical contract under which a request's output is
unchanged by the other requests with which it is batched.

**Block table** — A mapping from logical sequence blocks to physical cache
blocks.

**Cache-aware routing** — Request placement that considers reusable state as
well as load and other constraints.

**Chunked prefill** — Processing a long prompt across several engine steps so
other work, especially decode, can advance between chunks.

**Closed-loop load generation** — A workload in which a client waits for a
response before issuing more work. Offered load falls when the server slows.

**Collective** — A communication operation involving a group of ranks, such as
all-reduce, all-gather, reduce-scatter, or all-to-all.

**Compilation artifact** — Generated code, a graph, tuning result, or other
reusable output tied to an execution environment and configuration.

**Constrained decoding** — Generation that masks tokens according to a grammar,
schema, regular expression, or other allowed-output state.

**Context parallelism (CP)** — Partitioning sequence positions or context work
across ranks. The exact algorithm must be stated.

**Continuous batching** — Changing batch membership between inference
iterations so completed sequences leave and waiting sequences enter.

**Control plane** — Components that place, route, scale, and recover work across
engines rather than executing the current model step.

**Copy-on-write** — Sharing immutable state until a writer needs to modify it,
at which point a private copy is created.

**CUDA Graph** — A recorded GPU operation graph that can be instantiated and
replayed with reduced launch overhead.

**Data parallelism (DP)** — Replicating a model or model component so ranks can
process independent request work. In MoE deployments, attention DP may compose
with shared expert parallelism.

**Data plane** — The request-critical path that schedules and executes current
work, manages its state, and produces output.

**Decode** — The autoregressive phase that adds output positions, usually one
per active sequence per ordinary engine step.

**Decode-context parallelism (DCP)** — Partitioning decode context or KV state
by sequence position and combining partial attention results.

**Disaggregation** — Placing model stages, such as prefill and decode, in
separate worker pools with explicit intermediate transfer.

**E/P/D** — Encoder/prefill/decode disaggregation.

**End-to-end latency (E2E)** — Time from request arrival at the measured service
boundary to final response completion.

**Engine step** — One scheduler decision and its corresponding model execution
and output update.

**Expert parallelism (EP)** — Distributing different MoE experts across ranks
and routing token representations to their owners.

**Expert-parallel load balancing (EPLB)** — Changing expert placement or
replication based on observed routing load.

**Goodput** — Completed work per time that satisfies a stated latency,
correctness, quality, and error contract.

**Graph bucket** — A captured or compiled execution shape selected to cover a
range of runtime batches, often with padding.

**Grouped GEMM** — Execution of several matrix multiplications, often with
different shapes, through one coordinated operation. Common in MoE layers.

**HBM** — High-bandwidth memory attached to an accelerator.

**Hierarchical cache** — A cache that places state across tiers such as GPU,
host memory, local storage, and remote storage.

**Inter-token latency (ITL)** — The time between consecutive visible output
tokens or stream events.

**JIT compilation** — Generating or specializing executable code at runtime.

**KV cache** — Persistent attention keys and values created from earlier token
positions. The term is sometimes used loosely for other model-specific sequence
state.

**KV connector** — An engine interface or implementation that moves or stores
KV state outside its local cache manager.

**Latency percentile** — A value below which a stated percentage of observations
falls within a defined population and window.

**Management plane** — Deployment and policy systems that change the service's
configuration, software, model, or capacity.

**Model runner** — The engine component that prepares device tensors and
invokes model code, kernels, graphs, and collectives for a scheduled step.

**MoE** — Mixture of experts, a model layer that routes each token to a subset
of expert networks.

**Multi-head latent attention (MLA)** — An attention architecture that stores
and operates on compressed latent representations rather than a conventional
full KV layout.

**NUMA** — Non-uniform memory access, in which CPU memory access cost depends
on the socket or node that owns the memory.

**Open-loop load generation** — A workload that sends requests according to an
external arrival process regardless of current server latency.

**P/D** — Prefill/decode disaggregation.

**Paged attention** — Attention over KV state stored in noncontiguous physical
blocks addressed through logical mappings.

**Pipeline parallelism (PP)** — Assigning ranges of model layers or stages to
different ranks and sending activations between them.

**Prefix cache** — Retained model state for a reusable beginning of an input.

**Prefill** — Processing input tokens or positions to produce the first output
and persistent state for later decode.

**Preemption** — Removing a running request from active execution to free
capacity, with later recomputation or state restoration.

**Prompt cache** — A broad term for reusable results derived from prompts. State
the representation: tokens, processed media, encoder features, or KV blocks.

**Radix cache** — A prefix cache indexed by a radix tree over token or other
discrete input sequences.

**Rank** — One member of a distributed process group, identified by an integer
within that group.

**RDMA** — Remote direct memory access, a family of mechanisms for moving data
between registered memory regions with reduced CPU involvement.

**Reduce-scatter** — A collective that reduces values and leaves a different
shard of the result on each rank.

**Request goodput** — Requests completed within a specified service and quality
contract per unit time.

**Roofline model** — A performance bound that compares peak compute with memory
bandwidth times arithmetic intensity.

**Router** — A control-plane component that assigns new requests or stages to
engines or worker pools.

**Scheduler** — The data-plane component that chooses which admitted work
advances in the next engine step.

**Sequence parallelism (SP)** — A family of methods that shard sequence-related
work or tensors. Define the specific data layout and communication when using
the term.

**Service-level objective (SLO)** — A measurable target for latency,
availability, correctness, quality, or another service property.

**Session affinity** — Routing related turns or events to the same worker to
preserve local state.

**Speculative decoding** — Proposing future tokens with a cheaper method and
verifying them with the target model so one target step may advance several
tokens.

**State-space model (SSM)** — A sequence model that maintains recurrent state
instead of or alongside token-indexed attention state.

**Structured output** — Output constrained or interpreted according to a schema,
grammar, tool protocol, or parser contract.

**Tensor parallelism (TP)** — Sharding matrices and operations within model
layers across ranks, usually with frequent collectives.

**Time per output token (TPOT)** — Average post-first-token generation time per
additional output token. It can hide individual stream stalls.

**Time to first token (TTFT)** — Time from request arrival to the first visible
output token or equivalent event.

**Token budget** — A scheduler limit on total token positions processed in one
engine step.

**Topology** — The physical or logical arrangement of devices, links, ranks,
model stages, and state owners.

**Warm-up** — Deliberate execution before measured or user traffic to load
weights, allocate memory, compile kernels, tune implementations, and capture
graphs.

**Weight transfer** — A protocol that moves updated model parameters from a
trainer or source to inference ranks.

**Write-back cache** — A policy that copies modified or newly created state to a
lower tier later, often near eviction.

**Write-through cache** — A policy that writes state to a lower tier as it is
created or admitted to the upper tier.
