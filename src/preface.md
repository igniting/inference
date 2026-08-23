# Preface

## When a model becomes a system

Training gives a model its capabilities. Inference is where those capabilities
meet a user.

Every generated token, classified image, synthesized frame, or agent action is
an inference event. It arrives with a deadline, consumes memory and compute,
and competes with other work. A model can be excellent in an evaluation and
still make a poor product if its first response is slow, its stream stalls, or
its cost rises faster than demand.

This is why inference matters. Training happens periodically; inference happens
every time the product is used. Better inference does more than lower a bill.
It makes longer contexts practical, admits more concurrent users, shortens the
feedback loop of an interactive system, and allows capable models to run on a
wider range of hardware. The [2025 Stanford AI Index](https://hai.stanford.edu/ai-index/2025-ai-index-report)
reported that the cost of querying a system at roughly GPT-3.5 capability fell
more than 280-fold between November 2022 and October 2024. Gains of that scale
change which applications are possible—and cause demand to grow in return.

The difficult part is that a model does not run alone. A production request
passes through an API, a tokenizer or media processor, queues, a scheduler,
accelerator kernels, memory managers, and an output stream. At cluster scale it
also crosses routers, caches, interconnects, replicas, and failure boundaries.
The model defines the computation. The serving system decides when and where it
runs, which state survives, and whether the result reaches the user on time.

That serving system is the subject of this book.

## How inference became a systems discipline

The field did not begin with large language models. Early production serving
systems concentrated on loading model versions safely, exposing stable APIs,
and batching independent predictions. TensorFlow Serving, open-sourced in 2016
and described in a [2017 systems paper](https://research.google/pubs/tensorflow-serving-flexible-high-performance-ml-serving/),
made model lifecycle management and high-performance serving first-class
concerns.

The workload changed as the models changed. The
[Transformer](https://papers.neurips.cc/paper/7181-attention-isall-you-need.pdf),
introduced in 2017, made attention the foundation of a highly parallel model
architecture. Large generative models built on that architecture do not simply
run once per request. They first process an input and then repeatedly execute
the model to produce new tokens, retaining attention state between steps.
Requests have different input lengths, finish at different times, and grow
their memory footprint while running. A static batch is a poor fit for that
shape of work.

**A short history of generative-model serving.**

```blockdiag
flowchart LR
    A["2016–2017: model serving and Transformers"] --> B["2022–2023: continuous scheduling and paged KV memory"]
    B --> C["2024: structured programs and prefix reuse"]
```

In 2022, [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)
showed that a generative serving system could schedule at the granularity of a
model iteration instead of waiting for an entire request batch to finish. That
idea—now commonly called continuous batching—lets completed requests leave and
new requests join between decoding steps.

In 2023, the
[vLLM PagedAttention paper](https://doi.org/10.1145/3600006.3613165) identified
the key-value cache as a central memory-management problem. Its solution
borrowed the idea of paging from operating systems: request state could occupy
non-contiguous blocks and be shared safely, reducing waste and making larger
batches possible.

In 2024, [SGLang](https://mast.stanford.edu/pubs/sglang_efficient_execution_of_structured_language_model_programs/)
expanded the unit of optimization beyond a single prompt. Real applications
contain repeated prefixes, branching calls, tool interactions, constrained
outputs, and parallel generations. Its runtime used radix-organized prefix
state and structured-generation optimizations to execute those programs as a
whole.

These milestones changed the central question. Inference performance was no
longer only about making one model invocation faster. It was about coordinating
many evolving requests and their state across a finite machine.

## Why vLLM and SGLang

Papers explain a design by isolating its contribution. Production repositories
show what happens when that contribution must coexist with everything else:
API compatibility, model diversity, numerical formats, distributed execution,
hardware backends, observability, failures, and a changing user workload.

vLLM and SGLang are valuable because their code exposes two actively developed
answers to the same serving problem. Their schedulers, cache managers, model
runners, distributed layers, and tests turn abstract trade-offs into concrete
decisions. Studying both helps separate a durable principle from one project's
current implementation. Where they converge, there is usually a shared systems
constraint. Where they differ, there is usually a trade-off worth understanding.

This edition studies reproducible snapshots: [vLLM at commit
`5cecfc0`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e)
and [SGLang at commit
`e161bd1`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71),
with a manuscript snapshot date of August 23, 2026. The code will continue to
change. The book therefore organizes the implementation details around more
durable ideas: scheduling, state ownership, data movement, service contracts,
and evidence.

Chapter 0 begins with one request on one model worker. From there, the book
opens the engine, crosses accelerator and machine boundaries, and finally
confronts the operational and economic decisions of a production service. New
readers should start with Chapter 0 and continue in order. Experienced
practitioners can use the table of contents to enter at the problem they are
solving and consult the appendices only when they need a worksheet, reference,
or debugging procedure.

By the end, you should be able to take an unfamiliar inference system, draw its
critical path, locate the state and queues that govern it, predict where it will
saturate, and design a measurement that can prove your diagnosis wrong. That is
the craft this book aims to teach.
