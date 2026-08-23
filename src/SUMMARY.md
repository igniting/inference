# Summary

- [Inference Systems](index.md)
- [Preface](preface.md)

# Part I — The Inference Problem
- [Part I — The Inference Problem](part-1/index.md)

- [0. Your First Inference Request](part-1/00-first-request.md)
- [1. The Serving System: Decisions, State, and Ownership](part-1/01-serving-intelligence.md)
- [2. Workloads, SLOs, and Goodput](part-1/02-workloads-slos-goodput.md)
- [3. Model Topologies as Execution Graphs](part-1/03-model-execution.md)
- [4. Hardware Is a Topology](part-1/04-hardware-topology.md)

# Part II — Inside a Single Engine
- [Part II — Inside a Single Engine](part-2/index.md)

- [5. Anatomy of an Inference Server](part-2/05-engine-anatomy.md)
- [6. Scheduling the Decode Loop](part-2/06-scheduling.md)
- [7. Memory Management and Local Model State](part-2/07-kv-cache.md)
- [8. Kernels and Attention Backends](part-2/08-kernels.md)
- [9. Compilation and Graph Execution](part-2/09-compilation.md)
- [10. Quantization, Precision, and Determinism](part-2/10-quantization.md)
- [11. Speculative Decoding](part-2/11-speculation.md)
- [12. Adapter Serving and Multi-Tenant Customization](part-2/12-adapter-serving.md)

# Part III — Scaling Across Accelerators
- [Part III — Scaling Across Accelerators](part-3/index.md)

- [13. Parallelism as Data Movement](part-3/13-parallelism.md)
- [14. Serving Mixture-of-Experts Models](part-3/14-moe.md)
- [15. Stage Disaggregation: Encoder, Prefill, and Decode](part-3/15-disaggregation.md)
- [16. Hierarchical and Distributed Model-State Caching](part-3/16-distributed-cache.md)
- [17. Routing, Replication, and the Control Plane](part-3/17-routing.md)

# Part IV — Beyond Text-Only Decoding
- [Part IV — Beyond Text-Only Decoding](part-4/index.md)

- [18. Multimodal, Encoder, and Pooling Workloads](part-4/18-multimodal.md)
- [19. Diffusion, Image, Video, and World Models](part-4/19-diffusion-media.md)
- [20. Inference for Reinforcement Learning](part-4/20-rl-inference.md)
- [21. Interactive, Reasoning, and Agentic Systems](part-4/21-interactive-reasoning.md)

# Part V — Production Discipline
- [Part V — Production Discipline](part-5/index.md)

- [22. APIs, Streaming, and Structured Generation](part-5/22-apis.md)
- [23. Benchmarking and Performance Science](part-5/23-benchmarking.md)
- [24. Observability, Reliability, and Operations](part-5/24-operations.md)
- [25. Economics and Architecture Decisions](part-5/25-economics-architecture.md)
- [26. Security, Isolation, and Governance](part-5/26-security-governance.md)

# Appendices

- [A. Mathematical and Systems Notation](appendices/a-notation.md)
- [B. Hardware and Portability Reference](appendices/b-hardware-reference.md)
- [C. Reproducible Benchmark Cookbook](appendices/c-benchmark-cookbook.md)
- [D. Deployment Patterns and Decision Checklists](appendices/d-deployment-patterns.md)
- [E. Glossary](appendices/e-glossary.md)
- [F. Source and Reproducibility Ledger](appendices/f-source-ledger.md)
- [G. Worked Solutions](appendices/g-worked-solutions.md)
- [H. Optimization Migration Guide](appendices/h-migration-guide.md)
- [I. Production Debugging Playbook](appendices/i-debugging-playbook.md)
- [Research and Originality Policy](research-method.md)
