# Inference Systems

## Engineering Generative AI from Kernel to Cluster

This is the planning edition of a new, original book about production inference
systems. Its subject is broader than optimizing a single language model. Modern
serving systems coordinate heterogeneous model stages, accelerator memory,
compilers, kernels, distributed state, networks, schedulers, APIs, and
operational policy under changing workloads.

The book's central model is:

> An inference service turns a workload and a model topology into an execution
> plan, manages the plan's distributed state, and continuously adapts it using
> measurements from the production system.

That gives us six recurring questions:

1. What work arrives, and which service objectives matter?
2. What computation and persistent state does the model create?
3. How should the scheduler shape that work over time?
4. Where should computation, communication, and state live?
5. How should the control plane route, scale, and recover the service?
6. Which measurements demonstrate useful, correct, and economical output?

## Intended readers

The primary audience is engineers building or operating model-serving systems:
ML systems engineers, performance engineers, distributed-systems engineers,
and advanced practitioners moving from model APIs into inference
infrastructure. Readers should be comfortable with Python and basic deep
learning concepts. Accelerator programming and distributed systems are taught
as they become necessary.

## What makes this book different

The book is organized around durable engineering decisions rather than a
feature tour of one framework. vLLM and SGLang are important implementation
studies, alongside primary papers, hardware documentation, and reproducible
experiments. Their code is used to reveal design trade-offs—not to define a
framework-specific curriculum or declare a permanent winner.

Each chapter is expected to contain:

- a concrete request or system trace;
- an explicit state and data-movement model;
- at least one failure mode or misleading optimization;
- a measurement exercise with reproducible inputs; and
- implementation notes tied to dated source revisions.

The manuscript itself has not begun. The next step is to review the
[complete outline](outline.md), decide what to cut or combine, and then create
an evidence brief for every approved chapter.
