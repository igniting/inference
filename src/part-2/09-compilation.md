# 9. Compilation and Graph Execution

A decode step may finish on the GPU in less time than the CPU needs to prepare
and launch all of its kernels. When that happens, a faster kernel does not keep
the GPU busy. The empty spaces between kernels become the bottleneck.

Compilation and graph execution reduce that overhead by doing more planning
before the request arrives.

## Eager execution pays as it goes

In eager execution, the framework encounters operations and dispatches them at
runtime. This is flexible and easy to debug. It also repeats Python work,
operator selection, and launch preparation on every step.

A compiler captures a region of model computation and transforms it before
execution. It may fuse operations, generate specialized kernels, remove
redundant work, or choose layouts. PyTorch's official
[`torch.compile` documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
describes full-graph and region-based capture, dynamic shapes, specialization,
and debugging options.

Compilation has an up-front cost. If every request shape produces a new
specialization, the service can spend more time compiling than it saves. A
deployment needs a policy for dynamic dimensions and a cache for reusable
artifacts.

## CUDA Graphs capture launches

A CUDA Graph records a sequence of GPU operations and their dependencies, then
replays that sequence with much lower CPU launch overhead. NVIDIA's
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
separates graph use into definition, instantiation, and repeated execution.

Replay works because much of the operation structure is known. That creates
constraints. Kernel arguments and shapes must fit the captured graph. Memory
addresses often need to remain stable. Arbitrary host-side control flow cannot
appear inside a captured region. Some collectives or dynamic operations need
special handling.

An inference engine usually captures several shapes rather than one. At
runtime, it chooses a graph that can cover the active batch and pads or routes
unmatched work to eager execution.

## Padding versus too many graphs

Imagine capturing graphs for batch sizes 1, 2, 4, 8, 16, 32, and 64. A batch of
23 requests can use the size-32 graph with nine padded slots. Capturing every
possible size would avoid padding but consume more warm-up time and graph
memory.

This is a bucketing problem. Dense buckets reduce wasted work and increase the
number of artifacts. Sparse buckets reduce artifacts and increase padding.
Traffic distribution determines the right compromise.

The same issue applies to token counts, prefill chunks, speculative lengths,
multimodal shapes, and expert-routing capacity. “Enable CUDA Graphs” is only the
start of the execution plan.

## Full, piecewise, and breakable graphs

A full graph captures the entire model step. It offers a simple replay path but
fails when any region is too dynamic or incompatible.

A piecewise graph divides the model at deliberate boundaries. Static regions
use graph replay while dynamic operations run between them. A breakable graph
uses a similar idea but treats selected regions as allowed breaks within a
larger execution plan.

These approaches matter for modern serving workloads. An MoE router may produce
dynamic expert counts. A custom attention backend may not support capture for a
particular mode. A multimodal encoder may have dynamic dimensions. Keeping the
rest of the model in graphs preserves much of the launch benefit.

At the pinned snapshots, vLLM implements compilation passes and piecewise or
breakable graph machinery under
[`vllm/compilation`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/compilation).
SGLang contains full, piecewise, and breakable runners under
[`runner_backend`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/model_executor/runner_backend).
Their coexistence reflects a practical truth: serving graphs need controlled
escape routes for dynamic work.

## Warm-up is part of deployment

Compilation, autotuning, memory allocation, and graph capture often happen on
the first few representative shapes. Sending user traffic during this period
creates cold-start latency and can expose untested memory peaks.

A production warm-up should exercise the shapes, precisions, adapters,
attention backends, parallel groups, and structured-output paths expected in
traffic. It should also respect the deployment's memory ceiling. Capturing a
large graph after allocating the entire KV cache can fail even when both would
fit under a different reservation order.

Artifacts need version keys. Model weights, engine code, compiler version,
device architecture, kernels, and configuration can all make an old artifact
invalid.

## Diagnose before disabling

When graph or compiler performance disappoints, separate four cases:

- compilation time is appearing in the measurement;
- shapes are recompiling or missing graph buckets;
- execution falls back to eager mode;
- graph padding or memory constraints outweigh launch savings.

Use compiler logs, graph-dispatch metrics, and a GPU timeline. Compare cold,
warm, and steady-state runs. Record how many unique artifacts were created and
how often each one served real work.

For an experiment, choose a workload with variable batch sizes. Measure eager
execution, compiled eager execution, and graph replay. Report CPU preparation
time, GPU gaps, padding, warm-up time, graph memory, and end-to-end goodput. A
graph mode has succeeded only if it improves the service after its full cost is
included.

Compilation changes how operations run. Quantization, the subject of Chapter
10, changes the representation of the values they process.
