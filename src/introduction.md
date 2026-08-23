# Inference Systems

## Engineering Generative AI from Kernel to Cluster

A user sends 8,000 prompt tokens to a model and begins waiting. The GPU has
enough arithmetic capacity to answer quickly. Yet the first token arrives late,
the stream pauses twice, and a second request with the same long prompt repeats
nearly all the work.

Nothing in that story is explained by the model alone. The answer depends on a
scheduler deciding which tokens run, a memory manager finding room for their
state, kernels moving data through an accelerator, and a router deciding
whether reusable state is near the next worker. It also depends on what the
service promised the user. A fast average is little comfort if an interactive
response stalls every tenth token.

That is the territory of inference systems.

The central idea of this book is simple:

> An inference service turns a workload and a model topology into an execution
> plan, manages the plan's distributed state, and adapts it using measurements
> from the running system.

The sentence is compact; the machinery is not. A modern service may coordinate
tokenization, multimodal encoders, autoregressive decoders, diffusion stages,
accelerator memory, compiled graphs, expert networks, remote caches, and live
sessions. Improvements in one layer can move the bottleneck into another. A
larger batch raises throughput until it ruins latency. A remote cache saves
computation until transfer becomes slower than recomputation. A lower-precision
model saves memory until an important output changes.

## The six questions

We return to six questions throughout the book:

1. What work arrives, and which service objectives matter?
2. What computation and persistent state does the model create?
3. How should the scheduler shape that work over time?
4. Where should computation, communication, and state live?
5. How should the control plane route, scale, and recover the service?
6. Which measurements demonstrate useful, correct, and economical output?

These questions are more durable than any framework option. vLLM and SGLang
appear often because their implementations make current design choices
concrete. They are case studies, not the table of contents. Primary systems
papers, official hardware and software documentation, and reproducible
experiments provide the rest of the evidence.

## Who this is for

The book is written for engineers who build or operate model-serving systems:
ML systems engineers, performance engineers, distributed-systems engineers,
and practitioners moving from hosted model APIs into their own infrastructure.
You should be comfortable reading Python and recognize the broad shape of a
neural network. Accelerator programming and distributed execution are
introduced when they become necessary.

You do not need to memorize every kernel or parallelism scheme. The useful
skill is learning to trace a request: what computes, what moves, what persists,
what waits, and what evidence would prove an improvement.

## From one request to a fleet

Chapter 0 follows a single request through a single GPU — tokenization,
prefill, decode, and streaming — with concrete numbers on a real model. It is
the anchor: every later chapter adds complexity to the steps it introduces.

Part I establishes the workload, execution, and hardware vocabulary. Part II
opens a single engine and follows its scheduler, KV cache, kernels, compiled
graphs, numerical formats, decoding algorithms, and multi-tenant adapter
serving. Part III asks what changes when state and computation cross
accelerator or host boundaries. Part IV extends the model to images, audio,
video, reinforcement learning, interactive reasoning, and agentic waits. Part V
turns the mechanisms into API contracts, benchmark evidence, operating
practice, economic decisions, and security boundaries.

The appendices provide notation, hardware references and portability guides,
benchmark templates, deployment patterns, decision checklists, terminology,
source provenance, worked solutions, a phased optimization migration guide,
and a production debugging playbook.

Each chapter begins from a problem rather than a catalog of features. Most end
with an experiment or design exercise. Read the explanations first; return to
the source links when you want to see how a production implementation expresses
the idea.

If you are running an inference service today and need to solve a specific
problem, the [How to Read](how-to-read.md) page includes a problem-oriented
navigation table.
