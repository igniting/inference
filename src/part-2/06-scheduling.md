# 6. Scheduling the Decode Loop

At the beginning of a model step, an engine may have three kinds of work
waiting. Several conversations need one more decode token. A new request needs
to process a 12,000-token prompt. Another request has a deadline and higher
priority.

The GPU cannot execute a vague collection of requests. It needs a concrete
batch with valid memory for every position. Building that batch is the
scheduler's job.

## Why static batches waste work

In a static batch, the server groups requests and runs them together until all
finish. This works well when inputs and outputs have similar shapes. Text
generation is less cooperative.

One sequence may stop after five tokens while another continues for 500. The
finished sequence leaves an empty slot, but the batch remains alive for the
long request. Padding preserves a regular tensor shape while spending compute
and memory on positions that no user needs.

Iteration-level scheduling rebuilds membership between model steps. Finished
sequences leave and waiting sequences enter. This technique is commonly called
**continuous batching**. The
[Orca paper](https://www.usenix.org/conference/osdi22/presentation/yu) showed how
iteration-level scheduling improves transformer serving.

Continuous batching makes the GPU busier, but it also creates a fast control
loop. The engine must reconsider work and memory every step.

## A request count is not a work budget

Suppose the scheduler can admit 32 requests. That limit says little about the
next step. Thirty-two decoders need roughly one new token each. One large
prefill may need thousands of token positions.

Schedulers therefore use a token budget in addition to a sequence limit:

```text
scheduled tokens <= token budget
active sequences <= sequence budget
allocated state <= available capacity
```

Multimodal and speculative execution add more budgets. The engine may limit
encoder work, media items, draft tokens, or the number of requests using a
particular adapter.

Tokens are still an approximation. A prefill token with long-context attention
can cost more than a decode token. An MoE token may take a different path from
its neighbor. The budget is useful because it is cheap to compute, not because
all tokens are equal.

## The long-prompt problem

Return to the 12,000-token prompt. Processing it as one prefill operation may
occupy a long engine step. Every active conversation pauses while the GPU works
on the new prompt. Their average TPOT might remain acceptable, yet users notice
a large gap in streaming output.

**Chunked prefill** divides the prompt across several steps. The scheduler can
mix a chunk with decode work, allowing existing conversations to keep moving.
The [Sarathi-Serve paper](https://arxiv.org/abs/2403.02310) develops this idea as
a way to control interference between prefill and decode.

Smaller chunks protect decode latency but perform more scheduling and metadata
work. They may also lose some efficiency from large matrix operations. Very
large chunks recover that efficiency and recreate the stall. The right size
depends on the model, batch composition, parallel plan, and SLO.

This is a recurring pattern in inference systems: a parameter that looks like a
hardware tuning knob is also a policy about which user waits.

## Which request goes first?

First-come-first-served is easy to explain and usually fair by arrival time. It
can still let a large request block smaller ones when the work cannot be
chunked. Shortest-job policies improve average completion time but require an
estimate and can starve large requests. Priority queues protect important
traffic but need quotas or aging so low-priority work eventually runs.

Before choosing a queue policy, decide what fairness means. Equal request
starts, equal scheduled tokens, equal accelerator time, and tenant-weighted
shares produce different schedules. A token-based policy can remain unfair when
tokens have different costs because of context length, modality, or expert
routing.

Deadlines add another dimension. Work that cannot possibly finish before its
deadline may be better rejected than scheduled ahead of requests that could
succeed. This is one reason scheduling cannot replace admission control.

## What happens when memory runs out?

A running sequence consumes more state as it grows. Eventually the scheduler
may be unable to allocate the next block.

The cheapest response is to evict cached state that belongs to no active
request. If that is insufficient, the engine can wait, preempt a running
request, move state to another memory tier, or reject work.

Preemption frees capacity, but the evicted request loses time. If its state is
discarded, the engine must recompute the prefix later. If state is swapped, it
must move bytes out and back. Frequent preemption is often a sign that admission
allowed too many long-lived sequences or that the cache reservation left too
little headroom.

Victim choice changes the cost. A recently admitted request may have little
computed state to lose. A large request may free many blocks. A low-priority
request may be the correct product decision. The scheduler needs an explicit
policy rather than an accidental list order.

## Keeping the CPU ahead of the GPU

As GPU kernels become faster, the CPU work needed to prepare each step becomes
visible. Rebuilding tensors, copying metadata, processing old outputs, and
waiting for device results can leave gaps between GPU operations.

A **persistent batch** keeps stable slots for running requests and updates only
what changed. This reduces preparation and helps preserve fixed memory
addresses for graph replay.

An asynchronous scheduler goes further. While the GPU executes step `t`, the
CPU prepares step `t+1`.

```text
CPU: schedule t ---- schedule t+1 ---- schedule t+2
GPU:        execute t ----- execute t+1 ----- execute t+2
```

The overlap removes idle time, but the CPU is now making decisions with an
incomplete view. It may not yet know which speculative tokens were accepted or
which request just stopped. It must reserve memory conservatively and attach
versions to results. If a request is preempted, an old output may need to be
discarded. A block cannot be reused while an earlier step can still write it.

At the pinned revision, vLLM's
[`scheduler.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/core/sched/scheduler.py)
handles token budgets, preemption, encoder work, speculative lookahead, cache
connectors, and multiple in-flight batches in one scheduling path. SGLang's
[`scheduler.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/managers/scheduler.py)
and
[`overlap_utils.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/managers/overlap_utils.py)
show another approach to overlapping scheduling and execution.

The code is complicated because the interactions are real. Prefix hits,
chunking, speculative tokens, remote state, and asynchronous outputs all change
what “one more step” means.

## Admission protects the scheduler

A scheduler orders work that the service has accepted. It cannot rescue a
system that accepts more work than it can finish.

Imagine 120 requests arriving each second while the deployment can complete
100 within the SLO. Keeping the GPU full is not success; the queue grows by 20
requests every second. Eventually almost everyone waits too long.

Admission control uses the estimated work, queue, resident state, priority,
deadline, and downstream stage capacity to decide whether a request should
enter. Rejecting early may produce more goodput than accepting a request that
will time out. The rejection must also create backpressure. Automatic retries
without delay can turn overload into a larger burst.

## Worked example: one token budget, four requests

Give the scheduler 16 token slots per step. Request A arrives with a 24-token
prefill; B arrives beside it with four prompt tokens and needs eight output
tokens. If A consumes an unbroken prefill, B's interactive response waits even
though both fit within a few steps.

With an eight-token chunk limit, the first step can schedule eight tokens of A
and four of B. B can enter decode on the next step while A continues in bounded
chunks. Reserving decode slots prevents later prefills from breaking B's output
cadence.

The example is incomplete unless step duration depends on its composition. A
step with 16 prefill tokens and one with 16 decodes need not take the same time.
Use a measured lookup table keyed by decode batch and prefill tokens; otherwise
the simulator merely counts tokens.

## Practice: implement and explain a schedule

Simulate A `(arrival 0, prefill 24, output 4)`, B `(0, 4, 8)`, high-priority C
`(1, 8, 4)`, and D `(2, 20, 2)` under a 16-token step budget and 40 units of
live-state capacity. Compare FCFS with no chunking, eight-token chunks, and
priority plus aging.

Report each step's contents, TTFT, deadline-qualified goodput, preemptions, and
memory state. State when D should be rejected. A worked schedule and scoring
rule appear in [Appendix G](../appendices/g-worked-solutions.md#6-scheduler-simulation).

The scheduler can only make safe decisions if memory has clear ownership. The
next chapter examines the block manager and the reusable state behind it.
