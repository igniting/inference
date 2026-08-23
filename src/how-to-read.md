# How to Read This Book

The book supports several paths depending on your background and goal.

## If you are new to inference

Start with Chapter 0 ("Your First Inference Request"). It follows one prompt
through one GPU, end to end, with concrete numbers. Everything after Chapter 0
refers back to the steps it introduces. If any concept in a later chapter
feels unmoored, Chapter 0 is the anchor.

## The systems path

Read front to back starting from Chapter 0. Parts I and II build the
single-engine model; Part III extends it across accelerators and replicas;
Parts IV and V apply it to new serving loops and production operation. This
path suits engineers designing an inference platform.

## The performance path

Read Chapters 0, 2–4, then 6–16, then 23. Keep a real workload trace beside
you. For every mechanism, write down the expected change in compute, memory
traffic, communication, queueing, and quality before measuring it.

## The operations path

Read Chapters 0, 1, 2, 5–7, 15–17, 22–26, and Appendix I. This path emphasizes service
semantics, overload, distributed state, routing, observability, security, and
cost.

## The practitioner fast-track

If you already run vLLM or SGLang in production and want to optimize, read
Chapter 0 to calibrate, then jump to the problem-oriented table below. After
solving your immediate problem, read Appendix I (debugging) and Appendix D
(deployment patterns and decision checklists) to build a systematic approach.

## Problem-oriented navigation

If you have a specific problem to solve, start here instead of reading
front to back.

| Problem | Start with | Then read |
| --- | --- | --- |
| "My TTFT is too high under load" | Ch. 6 (scheduling, chunked prefill) | Ch. 7 (KV budget), Ch. 15 (P/D disaggregation), Ch. 23 (benchmark method) |
| "I need to serve a model that does not fit on one GPU" | Ch. 4 (hardware topology), Ch. 13 (parallelism) | Ch. 14 (MoE), Appendix B (memory worksheet) |
| "I want to add LoRA adapters in production" | Ch. 12 (adapter serving) | Ch. 7 (paged state), Ch. 17 (routing with adapters) |
| "Latency spikes during decode" | Ch. 6 (preemption), Ch. 8 (attention backends) | Ch. 9 (graph buckets and padding), Ch. 23 (isolating variables) |
| "I need structured JSON output reliably" | Ch. 22 (API contracts and grammar backends) | Ch. 11 (speculative verification) |
| "How do I benchmark properly?" | Ch. 23 (performance science) | Appendix C (benchmark cookbook), Ch. 2 (SLO definitions) |
| "Cache hit rates are low across replicas" | Ch. 16 (distributed caching) | Ch. 17 (cache-aware routing), Ch. 7 (prefix cache) |
| "I need to serve on AMD or non-NVIDIA hardware" | Appendix B (hardware and portability) | Ch. 4 (topology), Ch. 8 (backend selection), Ch. 9 (compilation) |
| "Deploying to production for the first time" | Ch. 0 (first request), Ch. 24 (operations) | Ch. 22 (API boundaries), Appendix D (deployment patterns) |
| "I need to serve multimodal or diffusion models" | Ch. 18 (multimodal), Ch. 19 (diffusion) | Ch. 15 (E/P/D disaggregation) |
| "How do I debug a serving issue?" | Appendix I (debugging playbook) | Ch. 24 (observability), Appendix D (decision checklists) |
| "Cost per request is too high" | Ch. 25 (economics) | Ch. 10 (quantization), Ch. 11 (speculation), Ch. 15 (disaggregation) |

Each row leads to a self-contained reading sequence. The first column
is the starting point; subsequent chapters fill in the mechanisms and
evidence the solution requires.

## A recurring worksheet

Use this table for every system discussed in the book. It restates the six
questions from the introduction as fields you can fill in for any system:

| Dimension | Questions |
| --- | --- |
| Workload | What arrives? With which length, modality, priority, and burst distribution? |
| Contract | Which latency percentile, quality, availability, and cost limits matter? |
| Model | Which stages, conditional paths, and persistent states exist? |
| Schedule | What is the next unit of work, and why is it chosen? |
| Placement | Where do weights, activations, caches, and queues live? |
| Movement | Which bytes cross memory or network boundaries per step? |
| Failure | What happens when capacity, a worker, or a transfer disappears? |
| Evidence | Which measurement would disprove the design assumption? |

## Notation and examples

Variables are introduced near their use and collected in Appendix A. Numeric
examples are intentionally small enough to inspect. They teach a method, not a
capacity promise. Hardware throughput, software support, and repository APIs
change; the source ledger pins the evidence used for this edition. Chapter 0's
reference card is the canonical definition of the Atlas planning constants.

Commands are illustrative unless a chapter labels them as tested. Never copy a
serving configuration into production without checking the model license,
engine revision, device support, exposed network interface, authentication, and
memory headroom.

## Reading the diagrams

Blue-outlined blocks represent components, stages, or owned state. Teal arrows
show data, control, or feedback movement; the nearby label says which when the
distinction matters. A decision diamond divides legal paths rather than ranking
them. Tables beneath diagrams make the comparison exact by naming the object,
constraint, failure, or measurement associated with each path.

Treat every diagram as a hypothesis about the system boundary. Real
implementations may combine boxes in one process or split one box across many
workers. The questions to preserve are who owns the state, which dependency
crosses the arrow, and which queue can delay it.

## How to use the exercises

The chapters follow one fictional service, Atlas, so the exercises accumulate
rather than reset. First read the worked example in the chapter. Attempt the
practice problem with the stated inputs and produce the requested artifact: a
trace, table, state machine, simulator result, test plan, or decision record.
Then compare your reasoning with [Appendix G](appendices/g-worked-solutions.md).

The solutions are not configuration recipes. They make assumptions visible,
show the calculation or invariant, and state what measurement could change the
decision. A different answer is stronger when it explains the same constraints
with better evidence.
