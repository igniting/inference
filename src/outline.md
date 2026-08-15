# Complete Book Outline

Working title: **Inference Systems: Engineering Generative AI from Kernel to
Cluster**

This is a structural blueprint, not manuscript prose. It deliberately starts
with workload and state, then moves through execution, distribution, emerging
model families, and production operation. The order is designed to teach a
method for reasoning about inference systems rather than cataloging framework
features.

## The book's through-line

Every serving design can be examined as a loop:

1. characterize the arriving work and its service objectives;
2. derive the model's computation, communication, and state;
3. construct a legal execution plan over a hardware topology;
4. schedule requests and manage state as conditions change;
5. observe useful output, cost, and failure; and
6. revise the plan or the control policy.

The same loop will be applied to language models, multimodal encoders,
mixture-of-experts models, diffusion pipelines, real-time media models, and
post-training rollouts.

---

# Part I — The Inference Problem

Part I establishes the language of the book. It replaces vague goals such as
“make inference fast” with explicit workloads, state, topology, and service
objectives.

## 1. Serving Intelligence

**Question:** What system exists between an incoming request and a useful model
output?

- Inference versus training: shared operations, different constraints
- The lifecycle of a request: ingress, preprocessing, execution, streaming,
  termination
- Data plane, control plane, and management plane
- Model execution as stages, not a monolithic forward pass
- Persistent, reusable, transient, and externalized state
- Online, offline, interactive, and rollout services
- The book's six-part reasoning model: workload, model topology, scheduler,
  execution plan, distributed state, measurement loop
- Why local optimizations can make the end-to-end service worse

**System investigation:** trace one chat-completion request through both vLLM
and SGLang at recorded commits, naming every queue, process boundary, state
owner, and synchronization point without comparing speed.

## 2. Workloads, SLOs, and Goodput

**Question:** What does “better” mean for a particular inference service?

- Requests, sessions, turns, sequences, tokens, frames, and samples
- Time to first token, inter-token latency, time per output token, and
  end-to-end latency
- Throughput, utilization, concurrency, and goodput
- Tail latency and why averages hide queueing failures
- Arrival processes, burstiness, correlations, and closed- versus open-loop load
- Input length, output length, modality, cacheability, and priority
  distributions
- Quality, correctness, availability, cost, and energy as first-class
  objectives
- Turning product promises into measurable service-level objectives

**Lab:** construct three synthetic workloads with identical token throughput but
different arrival and length distributions; show how they produce different
latency and goodput conclusions.

## 3. How Generative Models Execute

**Question:** Which properties of a model determine the shape of its serving
system?

- Tokenization, embeddings, transformer blocks, logits, and sampling
- Prefill and decode as distinct computational regimes
- Attention arithmetic, the KV cache, and sequence growth
- Dense, grouped-query, multi-query, and multi-head latent attention
- Local, sliding-window, sparse, and hybrid attention patterns
- Mixture-of-experts routing and conditional computation
- State-space, recurrent, and hybrid sequence models
- Encoder-decoder, multimodal, diffusion, and autoregressive media pipelines
- Separating model topology from the serving topology chosen to execute it

**Lab:** derive compute, memory, KV-state, and communication estimates for a
dense decoder, an MoE decoder, and one hybrid architecture.

## 4. Hardware Is a Topology

**Question:** Where can computation and state live, and what does it cost to
move them?

- Latency, bandwidth, capacity, and concurrency as different resources
- Compute-bound, memory-bound, and communication-bound regimes
- Roofline reasoning for prefill and decode
- Registers, on-chip memory, HBM, host memory, local storage, and remote storage
- CPUs as tokenizers, orchestrators, memory hosts, and sometimes executors
- PCIe, coherent memory, NVLink-class fabrics, and RDMA networks
- Collective-friendly islands, network rails, NUMA, and failure domains
- Mapping a logical parallel plan onto a physical topology
- Reading a hardware specification without turning it into a benchmark claim

**Lab:** draw the memory and interconnect topology of one real deployment, then
predict the bottleneck for short decode, long prefill, and KV transfer.

---

# Part II — Inside a Single Engine

Part II follows a request through one inference engine. It explains why the
scheduler, memory manager, kernels, compiler, and decoder must be designed as a
coupled system.

## 5. Anatomy of an Inference Server

**Question:** Which components cooperate to turn requests into streamed output?

- Protocol frontends, validation, and request normalization
- Tokenizers, processors, chat templates, and multimodal preprocessing
- Engine clients, schedulers, workers, model runners, and output processors
- Allocation, block tables, queues, and request state machines
- Sampling, detokenization, streaming, and cancellation
- Process architecture and the cost of crossing boundaries
- Synchronous, asynchronous, single-process, and multi-process execution
- Extensibility boundaries for models, kernels, devices, and plugins
- Comparing architectures by responsibility and state ownership

**Implementation study:** build parallel architecture maps for vLLM and SGLang,
then explain one design each makes easier and one it makes harder.

## 6. Scheduling the Decode Loop

**Question:** How should an engine choose the next useful unit of work?

- Static batching, iteration-level scheduling, and continuous batching
- Sequence slots versus token budgets
- First-come-first-served, priority, deadlines, and fairness
- Mixing prefill and decode under unequal costs
- Chunked prefill and head-of-line blocking
- Preemption, recomputation, swapping, and request retraction
- Persistent batches and overlap between CPU scheduling and GPU execution
- Admission control, backpressure, cancellation, and overload behavior
- Scheduler policies as product policy, not merely implementation detail

**Lab:** implement a small discrete-event scheduler simulator and compare FCFS,
priority, and chunked-prefill policies under bursts and mixed sequence lengths.

## 7. Memory Management and the KV Cache

**Question:** How can variable-length model state be managed without wasting the
accelerator?

- Deriving KV-cache size from model and sequence parameters
- Why contiguous allocation fragments under dynamic workloads
- Paging, blocks, block tables, and virtual-to-physical indirection
- Allocation, append, fork, copy-on-write, free, and eviction
- Prefix identity: tokens, adapters, modalities, positions, and model versions
- Prefix trees, hash chains, collision handling, and reuse policies
- Hybrid layer types and nonuniform state requirements
- Memory pressure, preemption, and graceful degradation
- Metrics that reveal fragmentation, churn, and misleading hit rates

**Implementation study:** trace block allocation and prefix reuse in vLLM and
SGLang; identify their correctness invariants and design a collision or stale
state test.

## 8. Kernels and Attention Backends

**Question:** When does a lower-level implementation change the performance
envelope of the whole engine?

- GEMM, attention, normalization, positional encoding, sampling, and MoE kernels
- Arithmetic intensity, memory traffic, occupancy, and launch overhead
- Tiling, fusion, persistent work, and split reductions
- Ragged batches, paged state, and variable sequence lengths
- Backend dispatch by device, dtype, shape, mask, and model architecture
- Architecture-specific kernels and portable fallbacks
- Autotuning, compilation caches, and cold-start costs
- Correctness tolerances, adversarial shapes, and numerical validation
- Why a faster microkernel may not improve end-to-end goodput

**Lab:** benchmark one operation at kernel, engine-step, and end-to-end levels;
account for any difference between the three speedups.

## 9. Compilation and Graph Execution

**Question:** How can dynamic serving workloads benefit from static execution?

- Eager execution, tracing, ahead-of-time compilation, and just-in-time
  specialization
- Operator fusion and compiler-generated kernels
- CUDA Graphs and eliminating repeated launch work
- Shape, address, control-flow, and collective constraints
- Full-graph, piecewise, and deliberately breakable execution
- Bucketing dynamic batches and sequence lengths
- Warm-up, graph capture, compilation caches, and deployment artifacts
- Interactions with adapters, quantization, attention backends, and parallelism
- Diagnosing recompilation, graph breaks, and silent fallback

**Implementation study:** map the compilation and graph modes exposed by vLLM
and SGLang, then design a workload that distinguishes warm, cold, and recaptured
execution.

## 10. Quantization and Numerical Behavior

**Question:** Which precision can be removed without violating the service's
quality and stability contract?

- Number formats, dynamic range, rounding, saturation, and accumulation
- Weight-only, weight-and-activation, KV-cache, attention, and communication
  quantization
- Per-tensor, per-channel, per-group, and token-dependent scaling
- Static calibration, online quantization, and model-native low precision
- Kernel and hardware support as part of the algorithm
- Memory savings versus dequantization and conversion overhead
- Accuracy, calibration, log-probability, and long-context evaluation
- Determinism, batch invariance, and reproducibility
- Building a quality-aware quantization decision table

**Lab:** evaluate two quantization strategies on memory, latency, throughput, and
task quality; identify a workload where the apparently smaller format loses.

## 11. Speculative and Constrained Decoding

**Question:** How can a server reduce serial decoding work or restrict outputs
without breaking distributional correctness?

- The serial dependency in autoregressive generation
- Draft-and-verify decoding and acceptance accounting
- Draft models, multi-token heads, feature-level predictors, and self-drafting
- N-gram, suffix, prompt-lookup, and retrieval-based proposals
- Tree proposals and parallel verification
- Adaptive proposal length, confidence thresholds, and cost models
- Grammar, JSON-schema, regex, and finite-state constraints
- Tool-call and reasoning parsers as part of the output contract
- Scheduler, graph-capture, batching, and KV-cache interactions

**Lab:** measure accepted tokens, verification cost, latency, and output
equivalence across easy and adversarial prompts; report when speculation hurts.

---

# Part III — Scaling Across Accelerators

Part III treats parallelism, disaggregation, caching, and routing as choices
about where computation and state live and when they move.

## 12. Parallelism as Data Movement

**Question:** How should a model be partitioned across a real hardware topology?

- Replication as the baseline form of data parallelism
- Tensor, pipeline, context, sequence, and decode-context parallelism
- Expert and attention-data parallelism
- Parameters, activations, tokens, experts, and KV state as shardable dimensions
- All-reduce, all-gather, reduce-scatter, all-to-all, and point-to-point traffic
- Latency bubbles, load imbalance, memory duplication, and communication overlap
- Composing parallel dimensions into a rank mesh
- Mapping logical axes onto nodes, switches, rails, and failure domains
- A decision procedure based on fit, workload, topology, and SLO

**Lab:** produce two legal parallel plans for the same model and cluster, predict
their bottlenecks, and validate the prediction with communication estimates.

## 13. Serving Mixture-of-Experts Models

**Question:** How does conditional computation change scheduling and
communication?

- Routing, top-k selection, capacity, and token imbalance
- Grouped GEMM and sparse expert execution
- Expert parallelism and all-to-all data movement
- Expert placement, replication, and redundant experts
- Prefill-optimized and decode-optimized communication modes
- Overlapping dispatch, compute, combine, and shared experts
- Dual-batch and two-batch overlap strategies
- Expert-parallel load balancing and placement updates
- Failure, stragglers, routing skew, and observability

**Implementation study:** compare the MoE execution paths and communication
backends available in vLLM and SGLang, using a common topology diagram and
traffic model rather than feature checkmarks.

## 14. Disaggregated Serving

**Question:** When should model stages run on different workers or different
hardware pools?

- Interference between prefill and decode
- Prefill/decode disaggregation and independent resource ratios
- Encoder/prefill/decode separation for heterogeneous pipelines
- KV and embedding transfer protocols
- Push, pull, rendezvous, and connector abstractions
- Static, conditional, and dynamically rebalanced disaggregation
- Queueing and goodput across coupled stage pools
- Stage-specific parallelism, kernels, and accelerator choices
- Failure recovery, retries, duplicate state, and partial availability

**Lab:** model a two-stage service as a queueing network and find the
prefill-to-decode ratio that maximizes goodput under two workload distributions.

## 15. Hierarchical and Distributed Caching

**Question:** How can expensive model state survive beyond one accelerator and
remain safe to reuse?

- Recompute versus retain versus transfer
- GPU, host memory, local storage, remote storage, and global cache tiers
- Block identity, prefix identity, namespaces, and model-version boundaries
- Radix indexes, hash metadata, directories, and cache events
- Admission, eviction, promotion, prefetch, and write-back policies
- Cache-aware routing and the tension between locality and load
- Consistency, failure, partial writes, and stale entries
- Multi-tenant isolation, privacy, and side channels
- Hit rate, useful bytes, transfer amplification, and saved compute

**Implementation study:** trace a reusable prefix across local and external
cache layers, then enumerate every validation needed before another request may
consume it.

## 16. Routing, Replication, and the Control Plane

**Question:** How should work be placed across replicas whose load and cached
state continually change?

- From process-local scheduling to cluster-level scheduling
- Load-only, locality-only, session-affine, and hybrid routing
- Data-parallel rank assignment and request ownership
- Queue visibility, telemetry delay, and stale decisions
- Admission control and global backpressure
- Autoscaling signals, startup lag, warm state, and hysteresis
- Elastic membership, draining, rolling deployment, and rebalancing
- Failure detection, retries, hedging, and duplicate work
- Separating routing policy from transport and runtime implementation

**Lab:** simulate a cache-aware router under skew and bursty load; find the point
where preserving cache locality reduces total goodput.

---

# Part IV — Beyond Text-Only Decoding

Part IV applies the same state-and-topology framework to workloads that do not
fit the classic decoder-only LLM loop.

## 17. Multimodal and Encoder-Heavy Serving

**Question:** How do preprocessing and encoders reshape the inference pipeline?

- Images, audio, video, documents, and heterogeneous request payloads
- Decode, validation, resizing, sampling, and feature extraction on CPUs and
  accelerators
- Vision and audio encoder execution
- Placeholder expansion and merging encoder outputs with language tokens
- Encoder batching, result caching, and cache identity
- Variable media sizes, frame counts, and memory pressure
- Encoder/prefill/decode scheduling and disaggregation
- Embedding, classification, reranking, and reward-model endpoints
- Protecting latency when preprocessing is untrusted or unbounded

**Lab:** profile a multimodal request from byte ingestion to first generated
token, then test which stage should be cached, batched, or disaggregated.

## 18. Diffusion, Image, Video, and World Models

**Question:** How should an engine serve iterative generative pipelines whose
state and batching rules differ from token decoding?

- Text encoders, denoisers or transformers, schedulers, decoders, and safety
  stages
- Iterative denoising versus autoregressive generation
- Compatible dynamic batching across steps, resolutions, and guidance modes
- Caching repeated computation in diffusion transformers
- Classifier-free guidance parallelism
- Tensor, sequence, ring, and Ulysses-style parallelism for media models
- Stage disaggregation and heterogeneous hardware assignment
- Causal video generation, sessions, and persistent state
- Quality-speed trade-offs and modality-specific evaluation

**Lab:** build a stage timeline for one image or video pipeline and compare
pipeline, tensor, and sequence-parallel placements under two interconnects.

## 19. Inference for Reinforcement Learning and Post-Training

**Question:** What changes when inference is one stage inside a learning loop?

- Rollout generation as a workload with waves, groups, and policy versions
- Co-located versus disaggregated training and inference
- Sleeping, waking, pausing, and relinquishing accelerator memory
- Partial rollouts, early stopping, cancellation, and resumption
- Weight transfer, refit, versioning, and atomic policy changes
- Invalidating compiled artifacts, prefixes, and KV state after an update
- Training-inference numerical mismatch and importance ratios
- Deterministic debugging and batch-invariant execution
- Fault tolerance and backpressure across the trainer-rollout boundary

**Implementation study:** trace one weight-update and rollout cycle through the
RL-serving facilities in vLLM and SGLang, marking all version and synchronization
boundaries.

## 20. Real-Time and Interactive Systems

**Question:** How does inference change when users continuously interact with a
long-lived session?

- Duplex voice, streaming video, agents, and event-driven applications
- Latency budgets across capture, transport, preprocessing, inference, and
  rendering
- WebSocket, streaming RPC, and bidirectional protocols
- Sessions, turns, interruptions, barge-in, and cancellation
- Partial inputs, incremental encoders, and incremental outputs
- Coordinating ASR, language, tools, TTS, vision, and media generation
- State ownership, migration, and session affinity
- Backpressure and graceful degradation under real-time deadlines
- Measuring conversational and perceptual quality, not token latency alone

**Lab:** construct a latency budget for a duplex voice agent, inject a slow tool
and a user interruption, and define the required cancellation semantics.

---

# Part V — Production Discipline

Part V turns mechanisms into a dependable service. It treats interfaces,
measurement, operations, economics, and security as architecture—not cleanup
work after the engine is fast.

## 21. APIs as Correctness Boundaries

**Question:** Which semantics must remain stable while the implementation
changes underneath them?

- Generate, chat, embedding, score, classify, rerank, and media endpoints
- Tokenization, chat templates, special tokens, and model-specific processors
- Sampling parameters, seeds, log probabilities, and reproducibility
- Streaming event order, usage accounting, and finalization
- Structured outputs, tools, reasoning content, and parser versioning
- Cancellation, deadlines, retries, idempotency, and duplicate requests
- Adapters, model selection, and per-request configuration
- Validation limits, authentication, rate limits, and tenant boundaries
- Compatibility tests and protocol conformance

**Lab:** create a protocol test suite for streaming cancellation and usage
accounting, then run it against two engine frontends.

## 22. Benchmarking and Performance Science

**Question:** How can an optimization claim be made reproducible and useful?

- Begin with a decision and hypothesis, not a benchmark command
- Kernel microbenchmarks, engine-step benchmarks, and end-to-end load tests
- Open-loop and closed-loop request generation
- Synthetic distributions, production traces, and trace anonymization
- Warm-up, compilation, cache state, and steady-state boundaries
- Percentiles, confidence intervals, goodput curves, and saturation
- Correctness, output quality, and semantic parity checks
- Profilers, timelines, counters, and bottleneck attribution
- Fair cross-engine comparisons and reporting negative results

**Lab:** write a benchmark card that another engineer can reproduce without
private context, including raw data and a falsifiable conclusion.

## 23. Observability, Reliability, and Operations

**Question:** How do operators know that the service is healthy, useful, and
recoverable?

- Metrics, structured logs, traces, and high-cardinality dimensions
- Queue, scheduler, allocator, cache, transfer, kernel, and output signals
- Readiness, liveness, warm-up, and model-loading states
- Tail-latency decomposition and critical-path traces
- Error budgets, overload, load shedding, and graceful degradation
- Failure injection for workers, networks, caches, and control planes
- Rolling upgrades, canaries, draining, and rollback
- Capacity planning, cold starts, disaster recovery, and dependency failure
- Runbooks that connect symptoms to state and data movement

**Lab:** design a dashboard and failure drill for a disaggregated service where
the cache remains reachable but one decode pool becomes partitioned.

## 24. Economics, Security, and Architecture Decisions

**Question:** How should a team choose and defend a production inference
architecture?

- Cost per request, output, good output, tenant, and SLO class
- Utilization, reservation, elasticity, and stranded capacity
- Energy, power limits, cooling, and carbon-aware scheduling
- Multi-tenancy, memory isolation, noisy neighbors, and quotas
- Prompt, output, adapter, cache, model, and supply-chain threats
- Untrusted model code, plugins, kernels, and serialization formats
- Privacy, retention, auditability, and regional constraints
- Managed service versus self-hosting versus hybrid ownership
- Architecture decision records and revisiting assumptions with evidence

**Capstone:** produce an architecture decision record for a specified workload,
including workload model, SLOs, model topology, hardware, parallel plan,
scheduler, state placement, failure policy, security boundaries, benchmarks,
and cost sensitivity.

---

# Appendices

## A. Mathematical and Systems Notation

- Shapes, bytes, FLOPs, bandwidth, latency, and utilization
- Queueing notation and percentile conventions
- Parallel rank and topology notation
- Symbols reused throughout the exercises

## B. Hardware and Communication Reference

- Accelerator memory hierarchy checklist
- Interconnect and collective-operation reference
- Topology discovery commands and interpretation
- Common bandwidth and synchronization traps

## C. Reproducible Benchmark Cookbook

- Environment and version capture
- Workload trace schema
- Raw result and benchmark-card templates
- Statistical checks and visualization conventions
- Quality and semantic-parity gates

## D. Deployment Patterns

- Single accelerator and single-node replicas
- Multi-node tensor or pipeline parallelism
- Expert-parallel MoE serving
- Prefill/decode and encoder/prefill/decode disaggregation
- Hierarchical KV caching
- Rollout workers integrated with post-training

## E. Glossary

- Canonical definitions and disambiguation of overloaded terms
- Framework-specific names mapped to system concepts
- Acronyms, units, and symbols

## F. Source and Reproducibility Ledger

- Repository commits examined for each chapter
- Primary papers and specifications
- Experiment manifests and raw artifacts
- Figure provenance and licenses
- Claim status: proposed, reproduced, reviewed, or superseded

---

# Editorial sequencing

The chapter order above is the reading order, not necessarily the drafting
order. After outline approval, the recommended production sequence is:

1. draft Chapters 1–7 to lock the conceptual vocabulary;
2. draft Chapter 22 and Appendix C to lock experimental standards;
3. investigate Chapters 8–16 directly against repository revisions;
4. draft Chapters 17–20 with model-family specialists and representative
   workloads; and
5. finish Chapters 21, 23, and 24 after the technical mechanisms have stable
   interfaces and failure models.

Before drafting, each chapter receives a one-page brief containing its learning
objectives, claims, evidence, code paths, figures, experiment, and explicit
out-of-scope list.
