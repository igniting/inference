# 9. Compilation and Graph Execution

Chapter 1 walked a service that lost. Graph replay cut launch overhead from
0.9 ms to 0.2 ms per step, yet users waited longer, because padding the batch
up to the captured size added 1.2 ms of GPU work to a step that had only been
5.2 ms long. Nothing in that story was a bug. Replay did exactly what it
advertised. The service regressed because the artifact was chosen without the
workload in hand.

That decision is this chapter's subject. A decode step may finish on the GPU
in less time than the CPU needs to prepare and launch all of its kernels. When
that happens, a faster kernel does not keep the GPU busy; the empty spaces
between kernels become the bottleneck. Compilation and graph execution attack
those spaces by doing more planning before the request arrives — but every
artifact they produce must be paid for in warm-up time, memory, and padding,
and each artifact is only worth its cost for the shapes that actually arrive.

## Visual map

**Compilation moves repeated host work into reusable artifacts.**

```blockdiag
flowchart LR
    E["Eager Python and dispatch"] --> O["Operation launches"]
    O --> G["GPU execution"]
    T["Captured or compiled graph"] --> R["Graph replay"]
    R --> G
    S["Runtime shape"] --> D["Artifact dispatcher"]
    D --> T
    D --> E
```

**Graph buckets trade artifact count against padding and fallback.**

```blockdiag
flowchart TB
    B["Requested batch size"] --> X{"Compatible captured bucket?"}
    X -->|Exact| R["Replay exact graph"]
    X -->|Larger bucket| P["Pad and replay"]
    X -->|None| E["Compile or eager fallback"]
    R --> M["Record dispatch outcome"]
    P --> M
    E --> M
```

**Mode dispatch tries the strictest key first and relaxes toward eager.**

```blockdiag
flowchart LR
    K["Batch descriptor"] --> D{"Dispatch"}
    D -->|"exact FULL key"| F["Full-graph replay"]
    D -->|"relaxed PIECEWISE key"| P["Piecewise replay"]
    P --> S1["Captured segment"] --> A["Attention boundary"] --> S2["Captured segment"]
    D -->|"no matching key"| E["Eager execution"]
```

| Artifact outcome | Immediate cost | Long-term risk | Metric |
| --- | --- | --- | --- |
| exact replay | low launch overhead | artifact memory | exact-bucket hit rate |
| padded replay | unused device work | latency at bucket gaps | padding ratio |
| eager fallback | repeated launches | CPU gaps | fallback rate |
| new compilation | warm-up and memory | artifact explosion | compile count and time |

## Eager execution pays as it goes

In eager execution, the framework encounters operations and dispatches them at
runtime. Each operation walks the same host path: the Python call enters the
dispatcher, the dispatcher selects an implementation — the registry work of
Chapter 8 — arguments are checked and marshaled, and a launch is issued. None
of this work depends on the request content. The same shapes arrive thousands
of times per second, and the host answers them identically each time.

The arithmetic explains why the gap exists at all. A large transformer runs
roughly a dozen kernels per layer; at eighty layers that is near a thousand
launches per step. At about a microsecond of host work per launch — a
reasonable planning figure, not a measurement — the host owes the GPU roughly
a millisecond per step, which is the same order as the 0.9 ms of launch gaps
Chapter 1 measured. The GPU work in a decode step does not shrink when the
batch is small, but the number of launches is fixed by the model, so the gap
hurts most exactly when utilization is already poor.

A compiler captures a region of model computation and transforms it before
execution. It may fuse operations, generate specialized kernels, remove
redundant work, or choose layouts. PyTorch's official
[`torch.compile` documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
describes full-graph and region-based capture, dynamic shapes, specialization,
and debugging options.

Compilation has an up-front cost. If every request shape produces a new
specialization, the service can spend more time compiling than it saves. A
deployment needs a policy for dynamic dimensions and a cache for reusable
artifacts — the same shape-keyed cache discipline Chapter 8's autotuner
already demands.

### Where a microsecond of dispatch goes

The per-launch host cost is not one number but a stack of small ones, and
knowing the stack tells you what compilation can and cannot remove. Assume a
declared breakdown for one eager operation: a few hundred nanoseconds for the
Python call frame and attribute lookups, a similar slice for the dispatcher's
pattern matching, a comparable slice for implementation selection and dtype or
device checks, and the remainder for argument marshaling plus the driver call
itself. Summed, the stages land near the microsecond figure used above — and
only the last stage is irreducibly necessary at step time.

Graph execution collapses the stack rather than shrinking each layer.
Implementation choice happened at capture; argument layout was fixed at
capture; the thousand launches became one replay call. What survives is the
host work *around* the graph — batch assembly, block tables, sampling
decisions — which is exactly the work Chapters 5 and 6 built processes and
overlap around. This is why graphs and scheduling overlap compose instead of
competing: replay removes launches from the critical path while overlap moves
the remaining host work off it.

## CUDA Graphs capture launches

A CUDA Graph records a sequence of GPU operations and their dependencies, then
replays that sequence with much lower CPU launch overhead. NVIDIA's
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
separates graph use into definition, instantiation, and repeated execution.
Capture runs the region once on a side stream while the driver records the
structure; instantiation turns the record into an executable; replay submits
the whole structure with a single launch call. A thousand launches become one.

Replay works because much of the operation structure is known. That creates
constraints, and each constraint exists for a concrete reason:

- **Shapes must fit the captured graph.** Kernel launch geometry is part of
  the record, so a replayed graph cannot grow its batch. Hence buckets.
- **Memory addresses must remain stable.** The executable bakes in pointers to
  its intermediate buffers. Paged KV cache helps here — Chapter 7's block
  tables give every sequence a stable home — but the activation workspace
  must persist for the process lifetime, which is why graph memory cannot be
  returned to the allocator between steps.
- **Host-side control flow cannot appear inside a captured region.** An early
  exit or a data-dependent branch executes on the CPU, and the CPU is not
  recorded. Dynamic decisions must move outside the graph or become
  device-side work.
- **Collectives need special handling.** Tensor-parallel all-reduces inside a
  captured region require communication libraries that support graph capture;
  a collective that synchronizes ranks from the host breaks the record.

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
Traffic distribution determines the right compromise. With power-of-two
buckets, the worst case is just under double: a batch of 17 replays the 32
bucket and pads 15 rows, 88 percent extra batch. The average case depends on
where traffic actually lands, which is why the practice exercise at the end of
this chapter hands you a distribution rather than a single number.

The same issue applies to token counts, prefill chunks, speculative lengths,
multimodal shapes, and expert-routing capacity. "Enable CUDA Graphs" is only the
start of the execution plan.

### Pricing the bucket set

The bucket set's price has three components, and the first is memory. Capture
allocates intermediate buffers through the caching allocator, and replay
requires those addresses to stay valid forever, so the pool persists for the
process lifetime. Assume the largest decode bucket's capture grows the pool by
2 GiB — a declared planning figure. If each bucket grew its own pool, five
buckets would hold 10 GiB hostage. Engines avoid this by capturing in a
deliberate order: vLLM's `get_capture_descs` in `vllm/v1/cudagraph_dispatcher.py`
sorts descriptors by `(num_tokens, num_active_loras)` **descending**, with the
stated intent of memory efficiency. The largest capture grows the pool once;
every smaller capture afterward allocates from the space already freed. Expect
the total to land near the largest bucket's footprint, not the sum.

The second component is warm-up time, and it multiplies faster than bucket
count suggests. Keys are the cross product of everything the graph depends on:
vLLM builds its keys with `product(cudagraph_capture_sizes, lora_cases)`, where
the LoRA axis is a single case without adapters, or `[0] + captured_counts`
when adapter counts are specialized. Assume 0.7 seconds per capture and four
LoRA cases over five sizes: twenty graphs, roughly fourteen seconds of warm-up
before the first request is admitted. Speculative decoding multiplies along a
different axis — `uniform_decode_query_len = 1 + num_speculative_tokens` means
each decode slot carries one token per draft plus the base token (Chapter 11),
so buckets must align to that stride.

The third component is the smallest: the instantiated executable itself is
MiB-scale, not GiB-scale. It still adds up across dozens of graphs, and
SGLang attacks exactly that term — the second guided reading below shows how.

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
[`vllm/compilation`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/compilation)
— `piecewise_backend.py`, `breakable_cudagraph.py`, and `partition_rules.py`
name the three concerns directly. SGLang contains full, piecewise, and
breakable runners under
[`runner_backend`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/model_executor/runner_backend)
as `full_cuda_graph_backend.py`, `breakable_cuda_graph_backend.py`, and
`tc_piecewise_cuda_graph_backend.py`, all behind one
[`BaseCudaGraphBackend`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/model_executor/runner_backend/base_cuda_graph_backend.py)
interface whose methods are `capture_one`, `can_run`, and `replay`. Their
coexistence reflects a practical truth: serving graphs need controlled
escape routes for dynamic work.

### Choosing where a graph may break

The boundary list is the real configuration decision in piecewise execution.
vLLM names it `splitting_ops`: the operations at which the compiler is allowed
to cut the model into captured segments. Attention sits on that list because
it is the one region whose shape genuinely varies — sequence lengths differ
per request, backends own their kernels (Chapter 8), and some modes refuse
capture outright. The dispatcher's constructor enforces the pairing at
startup with an assertion: piecewise cudagraph modes require that attention
is compiled piecewise or that breakable graphs are enabled, and the assertion
message prints the three settings involved (`cudagraph_mode`,
`compilation_mode`, `splitting_ops`) so a misconfiguration names itself.

Every additional break returns launch overhead to the step; every removed
break risks a capture failure or forces dynamic work into a padded shape.
MoE routing is the second common boundary candidate — expert counts are not
known until the router runs — so an MoE deployment may carry two break points
where a dense model carries one. The practical rule follows from Chapter 8's
compatibility thinking: put a boundary wherever a component's capture support
is uncertain, and nowhere else. Boundaries are load-bearing walls, not
decoration.

### Guided reading: how vLLM dispatches a step

The dispatcher that decides between those outcomes lives in
[`vllm/v1/cudagraph_dispatcher.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/cudagraph_dispatcher.py),
and its docstring states the contract plainly: the keys it holds "are the only
source of truth for valid cudagraphs that can be dispatched at runtime." The
wrappers that would replay a graph do not second-guess the choice; they trust
the dispatched mode or pass through to eager.

Three details reward close reading. First, keys are initialized late.
`initialize_cudagraph_keys` carries the comment that it "should be called only
after attention backend is initialized," because only then is the final
`CUDAGraphMode` known — Chapter 8's backend resolution feeds this dispatcher,
and until it resolves, the dispatcher's mode defaults to `CUDAGraphMode.NONE`.
Startup order is a dependency chain, not a formality.

Second, the bucket map is a precomputed round-up table.
`_compute_bs_to_padded_graph_size` builds a flat list from every batch size up
to the maximum; a size that lands exactly on a bucket maps to itself, and
everything between maps up to the next bucket. The same method then validates
`compile_sizes`: a compile size that padding would change raises a
`ValueError` telling the operator to use values from `cudagraph_capture_sizes`.
A shape that must not drift is refused at startup rather than silently
rounded.

Third, `dispatch` relaxes in one direction only. It rejects immediately when
`num_tokens` exceeds `max_cudagraph_capture_size`, returning `NONE`. It checks
the `FULL` mode with the exact descriptor first — the code comments that "FULL
mode needs exact num_reqs because FA3's scheduler_metadata computation depends
on it" — and only then relaxes the descriptor with `num_reqs=None,
uniform=False` to look for a `PIECEWISE` key, a search the docstring describes
as dispatching "a uniform batch to a graph that supports a more general batch."
LoRA adapter counts round up the same way: with specialization enabled,
`bisect_left` over `captured_lora_counts` finds "the smallest captured
`num_active_loras` that is >= the current." Strictest first, relaxation as
fallback, eager as the floor.

### Guided reading: SGLang graph backends and executable dedup

SGLang's `runner_backend` package separates the policy from the mechanism.
`BaseCudaGraphBackend` is a deliberately thin interface — `capture_one`,
`can_run`, `replay`, `cleanup` — and the three implementations differ in what
they capture, not in how callers invoke them.

The interesting mechanism is in
[`cuda_graph_dedup_mixin.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/model_executor/runner_backend/cuda_graph_dedup_mixin.py),
gated behind the `SGLANG_ENABLE_CUDA_GRAPH_DEDUP` environment flag. The
observation: many captured graphs are structurally identical — same kernels,
same launch geometry, same dependency order — differing only in which buffers
they read and write. Yet each instantiates its own executable. The mixin makes
structure, not shape, the sharing key. `graph_signature` walks the raw graph's
nodes and edges through the driver API, topologically sorts them (asserting
"CUDA graph contains a dependency cycle" if the sort fails to cover every
node), and returns the ordered node payloads plus the sorted edge list. A
kernel's payload is its name, grid dimensions, block dimensions, shared-memory
size, and launch attributes — deliberately excluding its arguments. Data
pointers are not part of the signature, which is precisely why two graphs over
different buffers can match.

Registration then exploits the match. The first graph of a signature
instantiates two executables: the live `graph_exec` and a `compat_exec` probe.
When a second graph with the same signature arrives, `register` calls
`cudaGraphExecUpdate` against the probe — proving compatibility without
disturbing the live executable — and adds the graph to the group. `seal` tears
the probes down at the end of capture and logs the payoff as "captured %d CUDA
graphs, deduped to %d execs."

Replay pays the flip side. If the requested graph is not the group's
`current_raw_graph`, replay first runs `cudaGraphExecUpdate` to repoint the
shared executable's parameters at this graph's buffers, then launches. Assume
that update costs on the order of tens of microseconds of host work: invisible
against a multi-millisecond step, but real, and incurred every time traffic
alternates between members of a group. The trade is one executable's memory
instead of N, bought with a small per-switch update. And the mechanism degrades
honestly: if the driver bindings are unavailable or the installed PyTorch lacks
`raw_cuda_graph`, `build_deduped_cuda_graph` returns `None` and the engine runs
"using plain executables" — an optimization that can fail without blocking
startup.

## Warm-up is part of deployment

Compilation, autotuning, memory allocation, and graph capture often happen on
the first few representative shapes. Sending user traffic during this period
creates cold-start latency and can expose untested memory peaks.

A production warm-up should exercise the shapes, precisions, adapters,
attention backends, parallel groups, and structured-output paths expected in
traffic. It should also respect the deployment's memory ceiling, and order
matters here: capturing a large graph after allocating the entire KV cache can
fail even when both would fit under a different reservation order, because
capture wants its pool contiguous in time if not in address space. The same
reasoning that makes Chapter 7 admit a sequence only when its blocks fit makes
a warm-up plan sequence its allocations deliberately.

Artifacts need version keys. Model weights, engine code, compiler version,
device architecture, kernels, and configuration can all make an old artifact
invalid. A cache keyed on too few of these serves stale code silently — the
failure mode is not a crash but a slow, unexplained regression, the same
signature Chapter 8's tuner cache guards against. Key on everything that
influenced the artifact, and treat an unkeyed influence as a bug in the key.

| What changed | Artifacts invalidated | Cheapest safe response |
| --- | --- | --- |
| Model weights | compiled code, captured graphs, tuned kernels | full re-warm |
| Kernel library or attention backend | tuned kernels, captured graphs | re-tune, re-capture |
| Engine or compiler version | compiled code, captured graphs | rebuild, compare timelines |
| Shape policy (`max_num_seqs`, spec tokens) | bucket keys and capture set | re-capture with new sizes |
| Device or driver | everything architecture-specific | rebuild from scratch |

## Diagnose before disabling

When graph or compiler performance disappoints, separate four cases:

- compilation time is appearing in the measurement;
- shapes are recompiling or missing graph buckets;
- execution falls back to eager mode;
- graph padding or memory constraints outweigh launch savings.

Each case has its own metric in the visual map's table: compile count and time,
exact-bucket hit rate, fallback rate, and padding ratio respectively. Use
compiler logs, graph-dispatch metrics, and a GPU timeline. Compare cold, warm,
and steady-state runs — a number that includes warm-up is answering a
different question than one that does not. Record how many unique artifacts
were created and how often each one served real work.

For an experiment, choose a workload with variable batch sizes. Measure eager
execution, compiled eager execution, and graph replay. Report CPU preparation
time, GPU gaps, padding, warm-up time, graph memory, and SLO-qualified
goodput. A graph mode has succeeded only if it improves the service after its
full cost is included — the lesson of Chapter 1's losing replay, applied as a
measurement discipline.

## Worked example: bucket 9 is really bucket 16

Suppose captured decode buckets are 1, 4, 8, 16, and 32. A batch of nine replays
the 16 bucket and executes seven padded slots. Walk the batch-of-eight case
first, where nothing pads. Eager execution spends 1.1 ms of CPU launch work;
graph replay spends 0.2 ms; dispatch and padding bookkeeping add 0.15 ms. The
net saving is 1.1 − 0.2 − 0.15 = 0.75 ms, so a 5.1 ms step becomes 4.35 ms.
Every term is measurable on a timeline, and the sum is the whole argument.

Batch nine is the interesting case. Replaying the 16 bucket still saves the
same 0.75 ms of host time, but now seven padded slots execute device work that
serves no request. The step loses exactly when that padded work exceeds 0.75
ms — and the padded work can be estimated from a slope rather than guessed.
Chapter 1's walked example measured padding at 1.2 ms for five added rows,
about 0.24 ms per row; at that slope, seven rows cost roughly 1.7 ms and the
replay loses by nearly a millisecond. But the slope is a property of the
workload, not a constant: in a weight-bound decode step the dominant traffic —
reading the weights — does not scale with batch size at all, so per-row cost
can be far lower than 0.24 ms. Measure the slope on your own step timeline
before predicting which side of 0.75 ms batch nine lands on.

Record requested shape, replayed bucket, padding ratio, fallback, and artifact
identity on every step. A histogram of requested batch sizes tells you whether
to add a bucket or accept eager execution for a rare gap. Compilation time is a
startup measurement, not something to hide inside or silently exclude from a
steady-state number.

## Practice: design the bucket set

Use the batch distribution `1: 8%, 2–4: 17%, 5–8: 31%, 9–16: 29%, 17–32: 15%`.
Compare eager, compiled eager, and graph replay with buckets 1, 4, 8, 16, and
32. Report cold start, CPU time, GPU gaps, padding, fallbacks, graph memory, and
goodput.

Propose one bucket change under a fixed graph-memory budget and explain which
traffic it helps. The worked analysis is in
[Appendix G](../appendices/g-worked-solutions.md#9-compilation-and-graph-buckets).

Compilation changes how operations run. Quantization, the subject of Chapter
10, changes the representation of the values they process.
