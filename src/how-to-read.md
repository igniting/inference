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

Read Chapters 0, 2–4, then 6–15, then 22. Keep a real workload trace beside
you. For every mechanism, write down the expected change in compute, memory
traffic, communication, queueing, and quality before measuring it.

## The operations path

Read Chapters 0, 1, 2, 5–7, 14–16, and 21–24. This path emphasizes service
semantics, overload, distributed state, routing, observability, security, and
cost.

## The practitioner fast-track

If you already run vLLM or SGLang in production and want to optimize, read
Chapter 0 to calibrate, then jump to the problem-oriented table below. After
solving your immediate problem, read Chapter 22B (debugging) and Appendix D2
(decision checklists) to build a systematic approach.

## Problem-oriented navigation

If you have a specific problem to solve, start here instead of reading
front to back.

| Problem | Start with | Then read |
| --- | --- | --- |
| "My TTFT is too high under load" | Ch. 6 (scheduling, chunked prefill) | Ch. 7 (KV budget), Ch. 14 (P/D disaggregation), Ch. 22 (benchmark method) |
| "I need to serve a model that does not fit on one GPU" | Ch. 4 (hardware topology), Ch. 12 (parallelism) | Ch. 13 (MoE), Appendix B (memory worksheet) |
| "I want to add LoRA adapters in production" | Ch. 11B (adapter serving) | Ch. 7 (paged adapter state), Ch. 16 (routing with adapters) |
| "Latency spikes during decode" | Ch. 6 (preemption), Ch. 8 (attention backends) | Ch. 9 (graph buckets and padding), Ch. 22 (isolating variables) |
| "I need structured JSON output reliably" | Ch. 11 (constrained decoding) | Ch. 21 (API contracts, grammar backends) |
| "How do I benchmark properly?" | Ch. 22 (performance science) | Appendix C (benchmark cookbook), Ch. 2 (SLO definitions) |
| "Cache hit rates are low across replicas" | Ch. 15 (distributed caching) | Ch. 16 (cache-aware routing), Ch. 7 (prefix cache) |
| "I need to serve on AMD or non-NVIDIA hardware" | Appendix B2 (hardware portability) | Ch. 4 (topology), Ch. 8 (backend selection), Ch. 9 (compilation) |
| "Deploying to production for the first time" | Ch. 0 (first request), Ch. 23 (operations) | Ch. 21 (API boundaries), Appendix D (deployment patterns) |
| "I need to serve multimodal or diffusion models" | Ch. 17 (multimodal), Ch. 18 (diffusion) | Ch. 14 (E/P/D disaggregation) |
| "How do I debug a serving issue?" | Ch. 22B (debugging guide) | Ch. 23 (observability), Appendix D2 (decision checklists) |
| "Cost per request is too high" | Ch. 24 (economics) | Ch. 10 (quantization), Ch. 11 (speculation), Ch. 14 (disaggregation) |

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
change; the source ledger pins the evidence used for this edition.

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
