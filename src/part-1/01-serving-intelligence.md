# 1. The Serving System: Decisions, State, and Ownership

Imagine that you send a question to a customer-support assistant. The answer
begins streaming half a second later and finishes a few seconds after that.
From the outside, one request went in and one answer came out.

Now picture the same assistant on a busy morning. The same question reaches
the same model, but the first word appears after four seconds, and the stream
stalls twice before finishing. Nothing about the model changed. The weights
are identical, the prompt is identical, and the hardware is identical. What
changed is the serving system around the model: how much work arrived, how it
was placed, and what had to wait.

Inside the service, much more happened than "run the model." A web server
checked the request and applied a chat template. A tokenizer turned text into
integers. A router chose a model replica. A scheduler found room beside other
requests already in progress. The model read weights and conversation state
from GPU memory one step at a time. Another component converted new token IDs
back into text and sent them over the network.

Any one of those steps can become the reason the assistant feels slow or
fails. Inference engineering is the work of understanding the whole path. This
chapter builds the map: the life cycle every request follows, the decisions
made at three different speeds, the state that must be protected along the
way, and the trap of improving one part while the service gets worse. Every
later chapter zooms into one region of this map.

We will build that map from running systems, not from an abstract architecture.
The two implementations followed throughout this book are vLLM and SGLang.
Their names and process boundaries differ, but both have to turn an incoming
request into scheduled GPU work and an ordered stream of output. Start with
their code paths; the vocabulary in the rest of the chapter will then name
things you have already seen.

## Meet the two engines

This edition studies fixed source snapshots so that a link continues to mean
the same thing as the prose beside it: vLLM at
[`5cecfc0`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e)
and SGLang at
[`e161bd1`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71).
Do not try to memorize either repository. Learn to recognize the same six
duties in both: accept a request, prepare it, admit it, schedule it, execute
it, and return its output.

### A first map of vLLM

The shortest useful route through vLLM begins at its asynchronous engine
interface, crosses a process boundary, and ends at the GPU model runner.

**A vLLM request crosses three major ownership boundaries.**

```blockdiag
flowchart LR
    A["OpenAI API and AsyncLLM"] --> B["EngineCore and Scheduler"]
    B --> C["Executor and GPUModelRunner"]
    C --> B
    B --> A
```

Use this source map on a first reading. The goal is to know where to resume
when a trace or metric points at one stage.

| Duty | Source anchor | What to notice |
| --- | --- | --- |
| Accept HTTP requests | [`entrypoints/openai/api_server.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/entrypoints/openai/api_server.py) | The protocol server creates and borrows an asynchronous engine client; HTTP handling is outside the engine core. |
| Prepare and track a request | [`v1/engine/async_llm.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/engine/async_llm.py) | `AsyncLLM.generate` creates a per-request output stream, processes the input, registers detokenization state, and submits the request. |
| Cross into the engine process | [`v1/engine/core_client.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/engine/core_client.py) | `EngineCoreClient` is the transport seam between asynchronous callers and the engine core. |
| Admit and advance work | [`v1/engine/core.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/engine/core.py) | `add_request` hands work to the scheduler; `step` schedules, executes, handles aborts, and applies results. |
| Choose tokens and KV blocks | [`v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/core/sched/scheduler.py) and [`v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/core/kv_cache_manager.py) | The scheduler spends a token budget; the KV-cache manager finds reusable blocks and allocates new ones. |
| Execute the model | [`v1/executor/abstract.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/executor/abstract.py) and [`v1/worker/gpu/model_runner.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/worker/gpu/model_runner.py) | The executor hides worker topology; `GPUModelRunner.execute_model` updates device-side request state and runs the selected model path. |

Follow one generation request in that order. The API layer calls
`AsyncLLM.generate`. That method's own documentation describes the handoff:
create an output stream, prepare the input, add detokenization state, then
submit to an `EngineCore` running separately. `EngineCoreClient` carries that
message across the process or transport boundary. `EngineCore.add_request`
places the request under scheduler ownership.

The repeated serving loop is visible in `EngineCore.step`. It asks
`Scheduler.schedule` for the next work, calls the executor's `execute_model`,
drains cancellation requests, and gives completed model output back to the
scheduler. The model runner updates request state and KV block tables before
dispatching the model. Results travel in the opposite direction: engine
outputs reach `AsyncLLM`'s background output handler, which feeds the stream
belonging to the original caller. A request therefore does not live in one
function. It changes owners at explicit seams.

### A first map of SGLang

SGLang exposes the same duties through a different split. Its
`TokenizerManager` is a substantial request-side manager, while its
`Scheduler` owns a long-running event loop and communicates with a tensor-
parallel model worker.

**An SGLang request moves from a frontend manager into a scheduler process.**

```blockdiag
flowchart LR
    A["HTTP server and TokenizerManager"] --> B["Scheduler and memory pools"]
    B --> C["TpModelWorker and ModelRunner"]
    C --> B
    B --> A
```

| Duty | Source anchor | What to notice |
| --- | --- | --- |
| Accept HTTP requests | [`entrypoints/http_server.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/entrypoints/http_server.py) | `generate_request` turns streaming results from the tokenizer manager into server-sent events and attaches cancellation behavior. |
| Validate, tokenize, and await output | [`managers/tokenizer_manager.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/managers/tokenizer_manager.py) | `TokenizerManager.generate_request` normalizes and validates input, tokenizes it, sends it onward, and waits on request-specific state. |
| Admit and schedule work | [`managers/scheduler.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/managers/scheduler.py) | `handle_generate_request` creates the scheduler's request object; `run_event_loop` repeatedly receives, batches, launches, and processes work. |
| Own reusable token state | [`mem_cache/radix_cache.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/mem_cache/radix_cache.py) and [`mem_cache/memory_pool.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/mem_cache/memory_pool.py) | The radix cache indexes reusable prefixes; the pools map requests and token positions to KV storage. |
| Execute and sample | [`managers/tp_worker.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/managers/tp_worker.py) and [`model_executor/model_runner.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/model_executor/model_runner.py) | `forward_batch_generation` builds a forward batch, invokes the model runner, and samples a next token on the final pipeline rank. |
| Convert tokens back to text | [`managers/detokenizer_manager.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/managers/detokenizer_manager.py) | The detokenizer manager maintains incremental decode state and sends text results toward the tokenizer manager. |

Trace the code from `http_server.generate_request`. It delegates to
`TokenizerManager.generate_request`, which normalizes the request, creates
request state, validates adapter selection, tokenizes input, sends the
tokenized object to the scheduler, and awaits responses. In the scheduler,
`handle_generate_request` constructs the internal `Req` object. The event loop
then receives pending work, chooses a batch, calls `run_batch`, and processes
the result.

The model-facing half begins in `TpModelWorker.forward_batch_generation`.
It constructs a `ForwardBatch`, calls `ModelRunner.forward`, and samples when
the worker owns the final pipeline stage. Back in the scheduler,
`process_batch_result` updates request progress and publishes a load snapshot
that a router can consume. Output token IDs pass through
`DetokenizerManager` before the frontend yields text to the HTTP stream. As in
vLLM, cancellation, memory release, and output delivery cross several owners;
closing the network connection cannot safely erase them all at once.

### The same duties, different boundaries

The comparison is more useful than either directory tree alone:

| Serving duty | vLLM | SGLang |
| --- | --- | --- |
| Request-side orchestration | `AsyncLLM` | `TokenizerManager` |
| Admission and repeated step | `EngineCore` plus `Scheduler` | `Scheduler` event loop |
| Prefix and KV ownership | `KVCacheManager` and block tables | radix cache and memory pools |
| Model execution | executor plus `GPUModelRunner` | `TpModelWorker` plus `ModelRunner` |
| Incremental output | async output handler and detokenizer state | `DetokenizerManager` and tokenizer-manager state |

Neither arrangement is the universal architecture. The important fact is
that both must assign the same duties and preserve state while ownership
changes. The rest of this chapter gives those duties portable names. Later
chapters return to these exact files and descend one level at a time.

## From model call to serving system

It is useful to begin with a simple request life cycle:

**One request crosses several queues and state owners.**

```blockdiag
flowchart LR
    A["Client"] --> B["API and validation"]
    B --> C["Router"]
    C --> D["Engine queue"]
    D --> E["Scheduler"]
    E --> F["Model runner"]
    F --> G["Output stream"]
    G --> A
    H["KV and session state"] <--> E
```


```text
receive -> validate -> prepare -> wait -> execute -> stream -> finish
```

Each stage has its own concern, and each can fail independently of the model.

Receiving and validating establishes what the request even is: which model,
which limits apply, whether attachments parse. Doing this before any
expensive work means a malformed request costs microseconds, not GPU time.
Preparation turns the caller's text into the model's input: a chat template
inserts role markers, and a tokenizer produces integer IDs. Given fixed
tokenizer and template revisions, this stage is deterministic, which is why
later chapters treat those revisions as part of the served contract.

The waiting stage matters because production requests rarely get a GPU to
themselves. A request that would execute in 200 ms may spend a second in a
queue ahead of it. It helps to notice that there is rarely one queue. An
edge queue forms in front of validation, an engine queue forms in front of
admission, and a distributed deployment adds queues between stages — a
completed prefill may wait for decode capacity on another worker. Queue
position is decided by admission and scheduling policy, which is why Chapters
6 and 16 own most of the latency story, and why a trace that records only
one queue length explains only part of every wait.

Execution may happen over hundreds of short model steps rather than one call.
The shape of the work also changes along the way. For a language model, the
initial prompt is processed as a group during **prefill**: a 1,000-token
prompt means roughly 1,000 positions computed together. New output tokens are
then produced one at a time during **decode**: each step adds a single
position per active conversation. Prefill and decode use the same weights but
stress the hardware differently, and Chapter 3 builds the machinery for
saying precisely how.

Streaming makes partial progress visible while decode continues, and
finishing releases request state or retains some of it for future reuse.
Retention is a decision, not an accident: kept state can serve the next turn
or the next user, and Chapter 7 formalizes when reuse is safe.

A vision-language request adds image decoding and a vision encoder before the
language model. An image-generation request runs a denoising model many times
instead of producing tokens at all. The stages survive, but their costs move.
For this reason, a request is better understood as a small workflow than as
one model invocation.

The precise single-request timeline belongs to Chapter 0. At this level, retain only the ownership map: the API owns validation, the router owns placement, the engine owns admitted work, and the output path owns ordered delivery. Load changes the time spent at those boundaries without changing the model itself. Chapter 2 names the resulting latency populations; Chapters 5 and 6 implement the engine boundaries and scheduling decisions.

## Three kinds of decisions

As the workflow runs, the service makes decisions at different speeds.

**The three planes operate at different time scales but share evidence.**

```blockdiag
flowchart TB
    M["Management plane: deploy and configure"] --> C["Control plane: place and recover"]
    C --> D["Data plane: schedule and execute"]
    D --> T["Metrics, logs, and traces"]
    T --> C
    T --> M
```

The first diagram follows one request left to right. Each arrow crosses an
ownership boundary: the client owns nothing after send, the API owns the
validated request, the router owns the placement decision, and the engine
owns everything from admission onward. The state box hangs off the scheduler
rather than the model runner because conversation state outlives any single
step; the runner borrows it for the duration of one batch.

The second diagram lifts the view. Requests flow through the data plane,
while measurements flow back out of it. When something breaks, two questions
come before any fix: which plane owns the failing decision, and which kind of
state was being read or written?

| Plane | Typical decision | State consulted | Decision cadence |
| --- | --- | --- | --- |
| Data | next token batch | request and block tables | every engine step |
| Control | destination replica | queues, locality, health | every request or event |
| Management | release and capacity | versions, policy, demand | minutes to days |


The **data plane** makes immediate decisions about current requests. It
chooses the next batch, allocates memory, launches model work, samples
outputs, and streams events. These decisions happen many times per second,
and their inputs must already be in memory: a data-plane decision that waits
on a network call stalls every conversation sharing the step.

The **control plane** decides where work should go. It routes requests
between replicas, tracks which prefixes are cached, changes membership when a
worker fails, and tells overloaded services to stop accepting more traffic.
Its view is broader than one model step, but it still reacts to live
conditions, usually within milliseconds to seconds.

The **management plane** changes the service itself. Deploying a new model,
rotating credentials, changing capacity, and rolling out a new engine version
belong here. These events are rare relative to request traffic, but they
redefine what correctness means for everything running underneath.

Separating these decisions prevents a common design mistake: making a
decision in the wrong place. A decode step that consults a global database
has moved a control-plane question into the data plane, and every user feels
the stall. Conversely, a deployment that ignores resident caches has moved a
correctness question into the management plane, where it is easiest to
miss. State created by old weights must not be reused with new weights, and
only coordinated rollout can guarantee that. The time scales differ, but
correctness connects the planes.

Most operational incidents are legible through this lens. A router that
keeps preferring a warm replica is a control-plane decision missing
data-plane evidence. A fleet that slowly drifts across versions is a
management-plane process missing enforcement. Naming the plane is often the
first step toward the fix.

What connects the planes is evidence. Metrics, logs, and traces are produced
by the data plane, aggregated for the control plane, and summarized for the
management plane; each consumer needs a different resolution of the same
events. This is why observability, Chapter 24's subject, is not a feature
added after the fact but the shared currency that lets three decision speeds
coordinate without sharing a fate.

## Follow the state

Component diagrams show where code runs. State tells you what the system
must protect.

Return to the support assistant. The model weights are long-lived and mostly
immutable. The request text, deadline, and generated tokens belong to one
request. Intermediate activations exist for only part of a model step. The
attention keys and values created from the conversation may live for the
whole request and remain useful for the next turn. Queues, worker health, and
cache locations describe the service as a whole.

These lifetimes suggest five broad categories:

| State | Examples | Typical lifetime |
| --- | --- | --- |
| Model | weights, tokenizer, compiled graphs | deployment or model version |
| Request | input, output, deadline, parser and random state | one request or session |
| Execution | activations, workspaces, collective buffers | part of a step |
| Reusable | KV blocks, encoder outputs, processed media | beyond one request |
| Service | queues, membership, routing and cache metadata | continuously changing |

For any important state object, ask who creates it, who may change it, how a
consumer recognizes the correct version, and what happens when the owner
fails. Those four questions uncover many bugs before code does.

### The four questions applied to one object per category

The questions earn their keep on concrete objects. Take one representative
from each category.

The **weight files** (model state) are created by the deployment pipeline and
changed only by a management-plane rollout. Consumers recognize the correct
version because the version travels with the deployment, not because a
worker happens to hold recent bytes. If the owning worker fails, recovery
must reload the same version; a replacement worker that fetches "current"
weights after a mid-rollout crash can join the fleet holding different
weights than its peers.

The **request record** (request state) is created at admission and mutated by
the data plane as steps complete. Its identity is the request ID, and every
downstream event must carry that ID or it cannot be attributed. If the
engine crashes, the record dies with it: the client experiences a disconnect,
and whether a retry is safe becomes an API contract question, which Chapter 22
develops.

An **activation workspace** (execution state) exists for part of one step and
is owned by the model runner. Nobody else may read or write it, and it is
freed when the step ends, success or not. Memory leaks in serving systems
usually mean someone broke this lifetime: a buffer allocated per step but
released conditionally.

A **KV block** (reusable state) is created during prefill and may outlive its
request. Identity is subtle: a block can be shared by branches of one
conversation, and whether it can be shared across requests depends on
content and context, which Chapter 7 owns. Physical memory is freed only when
the last reference disappears. The classic failure is freeing a block while a
GPU step still reads it, which corrupts an unrelated request's output.

A **queue entry** (service state) is owned by the control plane and changes
continuously. Its characteristic failure is staleness: a dead worker remains
listed as healthy until a check fires, and the router keeps sending traffic
to a destination that cannot answer. Freshness requirements, not importance,
distinguish service state from model state.

Cancellation shows how the categories interact under pressure. Closing the
network stream does not erase work already scheduled on a GPU. The service
must stop scheduling future steps, decide what to do with work already in
flight, release memory exactly once, and send a final protocol event if the
connection still exists. Each obligation lands on a different category:
future steps touch request state, in-flight work touches execution state,
memory release touches reusable state, and the final event touches the
request record again. Cancellation is therefore a state transition, not
merely an HTTP feature.

## The execution plan

An inference service needs a plan for placing and advancing work. The plan
answers questions such as these:

- How many requests can share a step? This sets throughput against
  per-request latency and is answered by the scheduler every step.
- Which devices hold each part of the model? A capacity and topology
  question, fixed at deployment and revisited in Chapters 4 and 12.
- How much memory is reserved for request state? Reservation protects
  interactive traffic from bulk work, and getting it wrong shows up as
  preemption storms.
- Which input shapes use compiled graphs? Capture trades flexibility for
  launch cost, and the shapes that actually arrive decide whether the trade
  pays.
- When should state be reused, moved, or discarded? Reuse saves computation
  but creates coupling between requests.
- Which traffic receives priority during overload? Overload is normal, not
  exceptional, and the priority policy is a product decision.

The best answers depend on the workload. A chat service with a long shared
system prompt benefits from prefix reuse. A batch summarization job may care
more about total completion time than first-token latency. A real-time voice
assistant values steady output and fast cancellation. There is no useful
configuration without a workload and a service objective.

The same plan questions, asked of three different services, produce different
plans — which is the strongest argument against copying configurations:

| Plan question | Interactive chat | Batch summarization | Voice assistant |
| --- | --- | --- | --- |
| requests per step | as many as latency allows | fill the batch | few, for steady cadence |
| memory reservation | protect interactive headroom | minimal; throughput first | strict, to avoid preemption |
| compiled graph shapes | common small batches | large fixed shapes | one shape, replayed constantly |
| reuse policy | aggressive prefix reuse | document-level reuse | conversation-local only |
| overload priority | paying or free tiers by class | job age | drop or degrade gracefully |

Reading the table by column shows each service committing to a coherent
posture; reading it by row shows that no single answer wins. A configuration
is only defensible next to the workload it serves.

Measurements complete the process. The service observes queues, latency,
throughput, cache use, failures, quality, and cost. Those observations show
whether the plan should change.

```text
workload and goals
       |
       v
model + hardware -> execution plan -> measured service behavior
                          ^                    |
                          +---- revise --------+
```

This feedback loop is the organizing idea for the rest of the book. Each
part narrows it: single-engine mechanics first, then distribution, then new
modalities, then the discipline of operating the loop in production.

## Why faster parts can create a slower service

Suppose a team captures the model in a GPU execution graph and saves a small
amount of launch overhead on every decode step. The captured graph expects a
fixed batch shape, so the engine pads small batches to a much larger size. At
low traffic, the GPU now performs enough extra work that users wait longer.

Both statements are true: graph replay made each prepared operation cheaper,
and the service became slower.

### Counting a padded replay

The paradox becomes clear with small numbers. Assume a measured step time of
5.2 ms of GPU work for a batch of three decoders running eagerly, plus 0.9 ms
of CPU launch gaps between kernels, for 6.1 ms wall time per step. Assume
also that the engine has captured graphs at batch sizes 1, 2, 4, and 8, and
that a captured step pads its tensors to the bucket size.

With three active requests, the engine selects the size-8 bucket. Padding
adds five idle rows to every batched operation. Suppose the padded step
performs 6.4 ms of GPU work, of which 1.2 ms belongs to the padding, and the
captured graph collapses launch gaps to 0.2 ms, for 6.6 ms wall time.

The local optimization worked exactly as advertised: launch overhead fell
from 0.9 ms to 0.2 ms. The service still regressed, because the step got
slower overall: 6.6 ms instead of 6.1 ms. Across a 200-token response, that
is roughly 100 ms of added time to first finish, paid by every user during
every quiet period, in exchange for launch savings that mattered only when
the GPU was already saturated. Whether the trade is right depends on the
traffic distribution, which is why Chapter 9 treats bucket selection as a
workload question and Chapter 23 insists on measuring it end to end.

The same tension appears in many forms. Cache-aware routing can overload the
replica with the best prefix. Large prefill chunks can improve GPU efficiency
while interrupting active decoders. Tensor parallelism can reduce arithmetic
per device while adding a network synchronization to every layer. A more
compressed weight format can save memory but use a slower kernel for the
shapes that actually arrive. Each of these reappears, with its own
arithmetic, in Chapters 16, 6, 12, and 10.

The lesson is not that local optimization is bad. It is that the unit of
success is the service objective under a realistic workload.

## Worked example: the cache hit that loses

Suppose a 6,000-token document question reaches a router. Replica A already
has 4,000 tokens of the document cached but has 450 ms of queued prefill
work. Replica B is idle and can recompute those 4,000 tokens in 240 ms. A
router that sees only cache locality sends the request to A and adds at
least 210 ms to the user's wait.

Walk the comparison. On replica B, the user waits the full recomputation:
240 ms of prefill before their own tokens begin producing output. On replica
A, the user first waits behind 450 ms of work that arrived earlier. Even if
the cached prefix reduced replica A's remaining work to nearly nothing, the
queue alone exceeds replica B's entire prefill by 210 ms. The cache saved
computation that was not the bottleneck; the queue was.

Cache value is therefore conditional on queue state, and a locality score
that ignores queues is not a conservative approximation — it is a different
decision. Chapter 17 builds routers that weigh both.

The useful trace is not merely `router -> worker`. It records the router's
queue estimate, matched-token estimate, decision time, and the worker that
became authoritative. At the worker it separates admission wait, allocation,
prefill, decode, and output buffering. With that trace, the wrong choice is
visible as data: a large matched-prefix count paired with a larger admission
wait than an idle alternative.

Ownership explains why the trace needs both ends. The request record lives
with the router and then the engine, while the KV blocks live on whichever
replica becomes authoritative. When the request is cancelled mid-flight,
cancellation must reach both owners without freeing blocks that a GPU step
still reads — the exact discipline the state categories imposed earlier.

This example gives the trace a purpose: explain why the apparently valuable
cache hit made the service slower. Chapter 2 will turn that observation into
a goodput and latency comparison.

## Practice: produce an ownership trace

Trace a 6,000-token document request with a 300-token output limit through edge
queue, validation, tokenization, routing, engine admission, prefix lookup,
prefill, decode, detokenization, and streaming. For every boundary, record the
queue, state owner, cancellation behavior, and one timestamp.

A useful artifact is a table with one row per boundary and one column per
recorded property, so that any row with an empty owner cell marks a state
object nobody is responsible for releasing. Then compare a replica with a
4,000-token match and 450 ms queue against an idle replica that recomputes
the prefix in 240 ms. State which replica you choose and which two metrics
would reveal a wrong choice in production. A worked answer is in [Appendix
G](../appendices/g-worked-solutions.md#1-request-trace).
