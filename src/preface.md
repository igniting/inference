# Preface

Inference looks deceptively small in a model diagram. Training produces
parameters; inference applies them. That description is mathematically sound
and operationally useless.

A production request may be tokenized on a CPU, admitted by a router that knows
where its prefix is cached, split between an encoder and a decoder, scheduled in
pieces alongside unrelated requests, executed through several compiled graph
variants, and streamed while its state migrates through multiple memory tiers.
For a mixture-of-experts model, every layer may redistribute tokens across a
fabric. For an interactive video model, the request may never really end. For a
reinforcement-learning rollout, the weights may change before the next batch.

The hard problem is coordination.

This book develops a way to reason about that coordination. Its central claim
is that an inference service is a stateful distributed system that repeatedly
turns a workload and a model topology into an execution plan. The scheduler
enacts that plan, the memory system preserves its state, the control plane
places it on hardware, and the measurement loop decides whether it works.

That framing prevents a common failure: optimizing the visible GPU operation
while degrading the service. A faster attention kernel can lose end-to-end if
it forces smaller batches. Prefix caching can lower compute and raise tail
latency if cache-aware routing creates hotspots. Disaggregation can isolate
decode latency and waste capacity if transfers or queues dominate. Quantization
can fit a model and still make it slower when the target shapes lack efficient
kernels.

## What this book is

This is an engineering book, not a framework manual. It uses vLLM and SGLang as
implementation studies because their code makes modern design choices concrete:
continuous scheduling, paged or radix-organized state, graph capture,
speculative decoding, expert parallelism, disaggregation, hierarchical caches,
multimodal execution, and post-training integration. The principles are meant
to survive their current APIs.

The implementation snapshot for this edition is:

- vLLM commit `5cecfc01375052698823fc401e31518fb32a981e`;
- SGLang commit `e161bd1265a0082478b7f1c09f224a52d315dc71`;
- manuscript snapshot date: August 15, 2026.

Features on a development branch are evidence of a design under active use or
investigation, not a promise of universal support. Every deployment must verify
its exact model, device, precision, backend, and release.

## What this book is not

It is not a rewritten edition of another author's work. Existing books can
reveal which questions readers care about, but the structure, explanations,
diagrams, examples, and exercises here were developed independently from
primary sources and implementation study. The project policy at the back of
the book explains how claims and sources are handled.

It is also not a list of optimization switches. Switches age quickly. The
questions behind them are durable:

- Which resource is saturated?
- Which state must persist, and who owns it?
- What work can legally share a batch?
- What moves when a model is partitioned?
- Where is the queue that determines tail latency?
- What correctness contract is being preserved?
- Which measurement would falsify the proposed improvement?

Chapter 1 turns these questions into an execution plan, and the worksheet in
How to Read This Book turns them into an evidence checklist.

## The contract with the reader

Each chapter begins with a service problem and ends with an investigation. The
mathematics is kept close to the decision it supports. Implementation details
are introduced through state ownership and data movement, then tied back to
observable behavior. When a claim depends on a particular repository revision,
the text says so.

You do not need to memorize every acronym. You should finish able to draw the
critical path of an unfamiliar inference system, locate its queues and state,
predict its likely bottleneck, and design a measurement that can prove you
wrong. That is the transferable craft of inference engineering.
