# Appendix B2. Hardware Portability Guide

This book uses NVIDIA GPUs and CUDA as its default platform because, at the
time of writing, most production inference systems run on them. But the
principles in the main text are not NVIDIA-specific. Many of them — scheduling
algorithms, cache management policies, routing logic, API contracts — are
device-agnostic by construction. Others — kernel selection, graph capture,
quantization formats — carry the same structural intent to every platform but
require different implementations.

This appendix maps the book's concepts to four non-NVIDIA platforms. It is not
a product comparison. It does not rank platforms or declare winners. Its purpose
is narrower: if you have understood a chapter's principle on NVIDIA hardware,
this appendix tells you where that principle transfers directly and where you
need to learn a platform-specific mechanism to apply it.

The platforms covered are AMD Instinct with ROCm, Google TPU with JAX/XLA,
Intel Gaudi with SynapseAI, and AWS Trainium/Inferentia with NeuronSDK. Each
is a first-class target in at least one major inference framework (vLLM,
SGLang, or both) and each has production deployments serving real traffic.


## Transfer table

The table below covers the book concepts that have platform-specific
implementations. For each concept, the "What changes" column describes the
structural difference — the thing you must account for when moving between
platforms.

| Book concept | NVIDIA (book's default) | AMD ROCm | Google TPU | Intel Gaudi | AWS Trainium | What changes |
|---|---|---|---|---|---|---|
| **Weights and memory budget (Ch. 3--4)** | 80 GB (H100), HBM3, ~3.35 TB/s | 192 GB (MI300X), HBM3, ~5.3 TB/s; 288 GB (MI355X) | 32 GB HBM (v5e), varies by generation; v6 increases capacity | 128 GB HBM2e per chip, ~3.7 TB/s | 32 GB HBM per core (Trn1), 2 cores per chip | Absolute capacity changes the KV budget arithmetic and the model-size boundary for single-chip serving. Bandwidth determines the decode roofline. The *method* from Chapter 4 — compute the ratio, find the knee — is unchanged. |
| **Attention backends (Ch. 8)** | FlashAttention-2/3, FlashInfer, xformers; Triton kernels | AMD Composable Kernel (CK), Triton for ROCm, hipBLASLt | Flash-like attention via Pallas/JAX custom kernels; Splash Attention | FusedSDPA (SynapseAI fused attention), custom Habana kernels | NeuronSDK fused attention operators | Backend *names* and *optimal tile shapes* differ. The selection logic from Chapter 8 — match the backend to the attention pattern, measure at step granularity, not kernel granularity — transfers directly. |
| **Graph capture / compilation (Ch. 9)** | CUDA graphs; optional torch.compile | HIP graphs (structurally identical API to CUDA graphs); torch.compile with ROCm backend | XLA compilation is mandatory, not optional. All execution is compiled. No eager fallback in the hot path | SynapseAI graph compiler; recipe-based graph capture | NeuronSDK compiler (neuron-cc); ahead-of-time compilation required | The chapter's framework — artifacts, warm-up cost, padding cost, bucket selection — applies everywhere. The key structural difference is optionality: on NVIDIA and AMD, graph capture is an optimization you can skip; on TPU and Trainium, compilation is the execution model. |
| **Quantization formats (Ch. 10)** | FP8 (E4M3/E5M2), INT8, INT4, AWQ, GPTQ, GGUF; wide kernel support | FP8 (OCP format on MI300X+), INT8, INT4; ROCm kernel coverage narrower than CUDA but expanding | BF16 native, INT8 via AQT; quantization choices constrained by XLA kernel availability | FP8 (E4M3), BF16, INT8; Gaudi-specific quantization recipes | BF16 native, INT8, FP8 support varies by generation; NeuronSDK quantization toolkit | Format *availability* varies. The chapter's principle — measure quality and throughput together, not separately — is platform-independent. The practical difference is that fewer quantized kernels exist on non-NVIDIA platforms, so the format-selection frontier is smaller. |
| **Parallelism and collectives (Ch. 12)** | NCCL over NVLink (intra-node) and RDMA (inter-node); NVSwitch for all-to-all | RCCL over Infinity Fabric (intra-node) and RDMA (inter-node); MI300X has high-bandwidth xGMI links | JAX pjit/shard_map with ICI (inter-chip interconnect) inside a TPU pod; DCN across pods | Habana Collective Communications Library (HCCL) over Gaudi internal mesh and scale-out NICs | NeuronSDK collective operations over NeuronLink (intra-instance) and EFA (inter-instance) | The *decision* of where to place tensor, pipeline, and expert boundaries is the same on every platform: map frequent collectives inside the fast fabric. What changes is the fabric topology, the library name, and the performance envelope of small versus large messages. |
| **Profiling tools (Ch. 22--23)** | Nsight Systems, Nsight Compute, DCGM, torch.profiler | ROCm Profiler (rocprof), Omniperf, Omnitrace, AMD SMI | JAX profiler, TensorBoard TPU plugin, Cloud TPU Profiler | Habana Profiler (hl-prof), SynapseAI Profiler, hl-smi | Neuron Monitor, Neuron Profile, neuron-top | The *methodology* from Chapter 22 — measure at the right boundary, isolate variables, control for warm-up — is universal. Only the tool names and output formats change. |


## What transfers without change

The following book concepts are implemented on the host CPU, in framework-level
logic, or at the API boundary. They do not touch device-specific code and
transfer to any platform without modification.

**Scheduling algorithms (Chapter 6).** Continuous batching, preemption
policies, priority queues, and admission control are CPU-side decisions. The
scheduler calls the model runner with a batch descriptor; it does not know or
care what device executes the batch. A scheduling algorithm written for an
NVIDIA deployment works identically on AMD or TPU hardware. The only indirect
effect is that device speed changes the time budget the scheduler has per step,
which may shift the optimal batch size or preemption threshold — but the
algorithm itself is unchanged.

**KV cache block management (Chapter 7).** Block tables, copy-on-write,
prefix-tree indexing, eviction policies, and the paged-memory abstraction are
all host-side data structures. The block manager allocates and tracks blocks;
a device-specific allocator provides the underlying memory. Swapping the
allocator is a clean interface change. The management logic, which is the
subject of Chapter 7, does not change.

**Routing and the control plane (Chapter 16).** Load balancers, session-affinity
routers, cache-aware routing, replica health tracking, and failover logic are
network-layer components. They operate on request metadata and backend health
signals, not on device APIs. A routing policy designed for an NVIDIA fleet
applies without modification to a heterogeneous fleet, provided the backends
expose the same health and capacity signals.

**API semantics (Chapter 21).** The OpenAI-compatible API contract — streaming
SSE, token-level callbacks, usage accounting, tool-call formatting — is
defined at the HTTP boundary. It is identical regardless of the device behind
the engine. Framework implementations (vLLM, SGLang) expose the same API
surface on every backend they support.

**Benchmarking methodology (Chapter 22).** The measurement discipline — control
variables, warm-up policy, percentile reporting, load-generation method — is
statistical methodology. It applies to any system that processes requests.
Only the profiling *tools* (the row in the transfer table above) are
platform-specific; the experimental design is not.

**Operational practices (Chapter 23).** Health checks, graceful drain, rolling
deployment, canary analysis, capacity planning, and incident response are
operational patterns. They depend on the control plane and monitoring
infrastructure, not on the device. An operational runbook written for an
NVIDIA deployment needs only tool-name substitutions (replace `nvidia-smi`
with `rocm-smi` or `hl-smi`) to apply elsewhere.


## What requires platform-specific work

These areas share the same *intent* across platforms but require different
implementations. When porting a deployment, budget engineering time for each.

### Kernel selection and fusion

Every platform has its own kernel library, and the set of available fused
operations differs. A fused attention-plus-RoPE kernel that exists on CUDA may
not have an equivalent on ROCm or SynapseAI. The consequence is not just a
name change — the optimal operation boundaries (which operations to fuse, which
to leave separate) may differ because the available fusions differ.

On NVIDIA, kernel selection is often implicit: FlashAttention, cuBLAS, and
Triton kernels are selected by the framework's backend dispatcher. On AMD,
the Composable Kernel library and ROCm's Triton fork fill the same role, but
the available tile shapes and fusion patterns may differ. On TPU, custom
kernels are written in Pallas (a JAX-native kernel language) and compiled
through XLA; there is no equivalent of loading a precompiled CUDA binary.
On Gaudi, SynapseAI provides a fixed set of fused operators, and custom
kernels use Habana's TPC programming model.

The engineering task: for each model architecture, verify that every operation
in the critical path has a performant kernel on the target platform. Measure
at step granularity, not kernel granularity, because a missing fusion may
shift work to a neighboring kernel in a way that isolated benchmarks miss.

### Graph and compilation artifacts

Chapter 9's framework — artifacts, warm-up cost, padding overhead, bucket
selection — applies everywhere, but the artifact format and lifecycle differ.

CUDA graphs and HIP graphs are structurally similar: capture a stream of
operations, replay them with updated parameters. The warm-up cost is the
capture time, and the padding cost is determined by the bucket strategy.
HIP graphs on ROCm follow the same API pattern and the same engineering
tradeoffs.

XLA compilation on TPU is a fundamentally different model. There is no eager
fallback in the serving path. Every distinct input shape triggers a
compilation (or retrieves a cached compilation). The warm-up cost is
compilation time, which can be significant for models with many shape
variants. The padding cost is determined by the shape-bucketing strategy,
just as with CUDA graphs, but the consequence of a cache miss is a full
recompilation rather than a fallback to eager execution. SGLang's JAX backend
manages this by maintaining a compilation cache with shape buckets tuned for
inference workloads.

NeuronSDK compilation is ahead-of-time: the model is compiled to a Neuron
Executable File Format (NEFF) before serving begins. Shape changes require
recompilation. This is the most constrained model — the artifact is fixed at
deployment time, and runtime shape flexibility depends entirely on the
bucketing strategy chosen during compilation.

SynapseAI graph compilation on Gaudi falls between these extremes. It
supports recipe-based graph capture with some runtime flexibility, but the
compilation cost is higher than CUDA graph capture and the shape constraints
are tighter.

### Quantization format support

The format-selection frontier from Chapter 10 is smaller on non-NVIDIA
platforms. FP8 support, which is broad on H100 and later NVIDIA hardware,
is available on MI300X and Gaudi 3 but with different kernel coverage. INT4
formats (AWQ, GPTQ) have mature CUDA kernels but may lack optimized
implementations on other platforms. On TPU, quantization is typically applied
through JAX's AQT (Accurate Quantized Training) library, which supports a
different set of formats than the CUDA ecosystem.

The engineering task: for each target precision, verify that (a) a kernel
exists, (b) it handles the model's shapes efficiently, and (c) the quality
impact matches what was measured on the reference platform. Do not assume
that a format that works well on CUDA will have equivalent kernel performance
on another platform.

### Communication libraries

NCCL, RCCL, HCCL, and NeuronSDK collectives implement the same collective
operations (all-reduce, all-gather, reduce-scatter, all-to-all) but with
different performance characteristics, topology awareness, and configuration
surfaces. The mapping of logical ranks to physical devices, which Chapter 12
treats as a critical performance decision, depends on the platform's
interconnect topology.

Key differences in practice:

- **Intra-node bandwidth.** NVLink and NVSwitch provide 900 GB/s per
  direction on H100 systems. AMD's Infinity Fabric on MI300X provides
  comparable bandwidth through xGMI links. TPU pods use a dedicated ICI
  mesh. Gaudi uses an internal mesh with 600 GB/s bisection bandwidth per
  node. The absolute numbers shift the crossover point between tensor
  parallelism and pipeline parallelism.

- **Inter-node transport.** All platforms support RDMA-capable networks, but
  the integration differs. NCCL auto-detects topology and selects algorithms;
  RCCL does the same on AMD systems. On TPU, inter-pod communication uses
  DCN (data center network) with different latency characteristics than ICI.
  On AWS, EFA (Elastic Fabric Adapter) provides the RDMA path for both
  Trainium and GPU instances, but NeuronSDK collectives are optimized
  specifically for EFA's topology.

- **Configuration.** Environment variables, topology files, and algorithm
  selection differ across libraries. A deployment that tunes NCCL with
  `NCCL_ALGO`, `NCCL_PROTO`, and topology XML needs equivalent tuning for
  the target platform's library.

### Memory management APIs

Chapter 7's block-management logic is device-agnostic, but the underlying
memory allocator is not. Each platform provides its own allocation,
deallocation, and transfer APIs. On NVIDIA, this is `cudaMalloc`,
`cudaMemcpy`, and the CUDA memory pool. On AMD, the HIP equivalents
(`hipMalloc`, `hipMemcpy`) are nearly identical. On TPU, memory management
is handled by the XLA runtime and is largely invisible to the application.
On Gaudi, SynapseAI manages device memory through its own allocation API.
On Trainium, NeuronSDK handles memory layout during compilation.

The practical consequence is that memory-pool tuning, defragmentation
strategies, and the interaction between graph pools and KV cache pools
(discussed in Chapters 7 and 9) require platform-specific configuration
even when the management policy is identical.

### Profiling toolchain

The methodology from Chapter 22 is universal, but the tools that implement
it are not interchangeable. Each platform's profiler exposes different levels
of detail, different visualization formats, and different overhead
characteristics.

| Platform | Timeline profiler | Kernel-level analysis | Device monitoring |
|---|---|---|---|
| NVIDIA | Nsight Systems | Nsight Compute | nvidia-smi, DCGM |
| AMD | Omnitrace | Omniperf | rocm-smi, ROCm SMI library |
| Google TPU | JAX profiler | Cloud TPU Profiler | TPU runtime metrics |
| Intel Gaudi | SynapseAI Profiler | hl-prof | hl-smi |
| AWS Trainium | Neuron Profile | neuron-top | Neuron Monitor, CloudWatch |

When porting a performance investigation, map each measurement from the
original profiling tool to the corresponding capability on the target
platform. Not every measurement has a direct equivalent — for example,
warp-level occupancy analysis (Nsight Compute) has no direct analog on
TPU, where the execution model is fundamentally different.


## Practical guidance

**Start with what transfers.** When evaluating a new platform, begin with the
device-agnostic layers: scheduling, cache management, routing, and API
compatibility. These are the largest fraction of the system by code volume
and they work immediately. The platform-specific layers — kernels, graphs,
quantization, collectives, profiling — are fewer components but require
deeper investigation.

**Measure, do not assume parity.** ROCm reaches 90--95% of H100 throughput
for standard inference workloads in mature frameworks, but this is an
aggregate statement. Individual operations — a specific attention pattern, a
particular quantization format, a given batch shape — may differ more. The
Chapter 22 methodology (measure at the right boundary, control variables,
report percentiles) is the tool for answering platform-specific performance
questions.

**Memory capacity changes the design space.** The MI300X's 192 GB of HBM
(and 288 GB on MI355X) changes which models fit on a single chip and how
much KV state can be resident. The memory-budget worksheet from Appendix B
applies unchanged — fill it in with the target platform's capacity and
bandwidth numbers. A model that requires tensor parallelism on 80 GB H100s
may fit on a single MI300X, eliminating communication overhead entirely.
Conversely, a platform with less memory per chip (32 GB on TPU v5e or
Trainium) may require parallelism for models that fit on a single H100.

**Compilation constraints are architectural, not incidental.** On TPU and
Trainium, compilation is not an optimization — it is the execution model.
This changes the deployment workflow: warm-up is compilation, not graph
capture; shape changes may require recompilation; and the bucket strategy
from Chapter 9 is not optional but mandatory. Plan for compilation time in
the deployment pipeline and in the capacity model.

**Framework support is the practical boundary.** The most important question
for a non-NVIDIA deployment is not whether a principle transfers (it does)
but whether the framework you use has implemented the platform-specific
layer. vLLM supports NVIDIA, AMD, TPU, Gaudi, and Trainium backends. SGLang
supports NVIDIA, AMD, and TPU (via its JAX backend). Check the framework's
backend maturity for your target platform — the presence of a backend does
not guarantee feature parity with the NVIDIA path.
