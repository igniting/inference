# Summary

- [Inference Systems](index.md)
- [Preface](preface.md)
- [Introduction](introduction.md)
- [How to Read This Book](how-to-read.md)

# Part I — The Inference Problem
- [Part I — The Inference Problem](part-1/index.md)

- [1. Serving Intelligence](part-1/01-serving-intelligence.md)
- [2. Workloads, SLOs, and Goodput](part-1/02-workloads-slos-goodput.md)
- [3. How Generative Models Execute](part-1/03-model-execution.md)
- [4. Hardware Is a Topology](part-1/04-hardware-topology.md)

# Part II — Inside a Single Engine
- [Part II — Inside a Single Engine](part-2/index.md)

- [5. Anatomy of an Inference Server](part-2/05-engine-anatomy.md)
- [6. Scheduling the Decode Loop](part-2/06-scheduling.md)
- [7. Memory Management and the KV Cache](part-2/07-kv-cache.md)
- [8. Kernels and Attention Backends](part-2/08-kernels.md)
- [9. Compilation and Graph Execution](part-2/09-compilation.md)
- [10. Quantization and Numerical Behavior](part-2/10-quantization.md)
- [11. Speculative and Constrained Decoding](part-2/11-speculation.md)

# Part III — Scaling Across Accelerators
- [Part III — Scaling Across Accelerators](part-3/index.md)

- [12. Parallelism as Data Movement](part-3/12-parallelism.md)
- [13. Serving Mixture-of-Experts Models](part-3/13-moe.md)
- [14. Disaggregated Serving](part-3/14-disaggregation.md)
- [15. Hierarchical and Distributed Caching](part-3/15-distributed-cache.md)
- [16. Routing, Replication, and the Control Plane](part-3/16-routing.md)

# Part IV — Beyond Text-Only Decoding
- [Part IV — Beyond Text-Only Decoding](part-4/index.md)

- [17. Multimodal and Encoder-Heavy Serving](part-4/17-multimodal.md)
- [18. Diffusion, Image, Video, and World Models](part-4/18-diffusion-media.md)
- [19. Inference for Reinforcement Learning](part-4/19-rl-inference.md)
- [20. Real-Time and Interactive Systems](part-4/20-realtime.md)

# Part V — Production Discipline
- [Part V — Production Discipline](part-5/index.md)

- [21. APIs as Correctness Boundaries](part-5/21-apis.md)
- [22. Benchmarking and Performance Science](part-5/22-benchmarking.md)
- [23. Observability, Reliability, and Operations](part-5/23-operations.md)
- [24. Economics, Security, and Architecture Decisions](part-5/24-economics-security.md)

# Appendices

- [A. Mathematical and Systems Notation](appendices/a-notation.md)
- [B. Hardware and Communication Reference](appendices/b-hardware-reference.md)
- [C. Reproducible Benchmark Cookbook](appendices/c-benchmark-cookbook.md)
- [D. Deployment Patterns](appendices/d-deployment-patterns.md)
- [E. Glossary](appendices/e-glossary.md)
- [F. Source and Reproducibility Ledger](appendices/f-source-ledger.md)
- [G. Worked Solutions](appendices/g-worked-solutions.md)
- [Research and Originality Policy](research-method.md)
