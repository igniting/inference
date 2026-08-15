# How to Read This Book

The book supports three paths.

## The systems path

Read front to back. Parts I and II build the single-engine model; Part III
extends it across accelerators and replicas; Parts IV and V apply it to new
serving loops and production operation. This path suits engineers designing an
inference platform.

## The performance path

Read Chapters 2–4, then 6–15, then 22. Keep a real workload trace beside you.
For every mechanism, write down the expected change in compute, memory traffic,
communication, queueing, and quality before measuring it.

## The operations path

Read Chapters 1, 2, 5–7, 14–16, and 21–24. This path emphasizes service
semantics, overload, distributed state, routing, observability, security, and
cost.

## A recurring worksheet

Use this table for every system discussed in the book:

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
