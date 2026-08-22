# Inference Systems

## Engineering Generative AI from Kernel to Cluster

An inference service turns a workload and a model topology into an execution
plan, manages the plan's distributed state, and adapts it using measurements
from the running system.

This book follows that sentence from a single decode step to a global fleet:
the scheduler that builds each batch, the memory that preserves its state, the
kernels and compiled graphs that execute it, the caches and control plane that
place it across machines, and the contracts, benchmarks, operations, and
economics that decide whether any of it serves a real product.

Implementation studies are pinned to vLLM commit
`5cecfc01375052698823fc401e31518fb32a981e` and SGLang commit
`e161bd1265a0082478b7f1c09f224a52d315dc71`.

[Start reading — Preface](preface.md)
