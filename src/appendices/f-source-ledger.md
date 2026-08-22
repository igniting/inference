# Appendix F. Source and Reproducibility Ledger

This edition distinguishes principles, implementation observations, and
measurements. A repository path proves that code exists at a snapshot; it does
not prove production readiness or performance on every platform.

## Edition snapshot

| Source | Revision or date | Role |
| --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | `5cecfc01375052698823fc401e31518fb32a981e` | implementation study |
| [SGLang](https://github.com/sgl-project/sglang) | `e161bd1265a0082478b7f1c09f224a52d315dc71` | implementation study |
| Manuscript | August 15, 2026 | claim cutoff |
| [Inference Engineering](https://www.baseten.co/inference-engineering/), Philip Kiely | supplied 259-page PDF, modified January 29, 2026 | editorial comparison only |

The supplied book informed the coverage audit and standards for approachability.
Its prose, diagrams, examples, analogies, and chapter sequence were not reused.

## Primary systems papers

- Ashish Vaswani et al.,
  [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
- Woosuk Kwon et al.,
  [Efficient Memory Management for Large Language Model Serving with
  PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
- Lianmin Zheng et al.,
  [SGLang: Efficient Execution of Structured Language Model
  Programs](https://arxiv.org/abs/2312.07104), 2023.
- Gyeong-In Yu et al.,
  [Orca: A Distributed Serving System for Transformer-Based Generative
  Models](https://www.usenix.org/conference/osdi22/presentation/yu), OSDI 2022.
- Amey Agrawal et al.,
  [Taming Throughput-Latency Tradeoff in LLM Inference with
  Sarathi-Serve](https://arxiv.org/abs/2403.02310), 2024.
- Yinmin Zhong et al.,
  [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large
  Language Model Serving](https://arxiv.org/abs/2401.09670), 2024.
- Pratyush Patel et al.,
  [Splitwise: Efficient Generative LLM Inference Using Phase
  Splitting](https://arxiv.org/abs/2311.18677), 2023.
- Ruoyu Qin et al.,
  [Mooncake: A KVCache-centric Disaggregated Architecture for LLM
  Serving](https://arxiv.org/abs/2407.00079), 2024.
- Lijie Liu et al.,
  [Preble: Efficient Distributed Prompt Scheduling for LLM
  Serving](https://arxiv.org/abs/2407.00023), 2024.

## Kernels, execution, and decoding

- Tri Dao et al.,
  [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135),
  2022.
- NVIDIA,
  [CUDA Programming Guide: CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).
- PyTorch,
  [`torch.compile` reference](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
- Yaniv Leviathan, Matan Kalman, and Yossi Matias,
  [Fast Inference from Transformers via Speculative
  Decoding](https://arxiv.org/abs/2211.17192), 2022.
- Guangxuan Xiao et al.,
  [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large
  Language Models](https://arxiv.org/abs/2211.10438), 2022.

## Adapter serving

- Ying Sheng et al.,
  [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285),
  2023.
- Chen Liang et al.,
  [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547),
  2023.
- Edward J. Hu et al.,
  [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685),
  2021.

## Parallel and MoE systems

- Mohammad Shoeybi et al.,
  [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model
  Parallelism](https://arxiv.org/abs/1909.08053), 2019.
- Deepak Narayanan et al.,
  [Efficient Large-Scale Language Model Training on GPU Clusters Using
  Megatron-LM](https://arxiv.org/abs/2104.04473), 2021.
- DeepSeek,
  [DeepEP](https://github.com/deepseek-ai/DeepEP), official expert-parallel
  communication implementation.

## Media and post-training

- Xinyin Ma, Gongfan Fang, and Xinchao Wang,
  [DeepCache: Accelerating Diffusion Models for Free](https://arxiv.org/abs/2312.00858),
  2023.
- Feng Liu et al.,
  [Timestep Embedding Tells: It's Time to Cache for Video Diffusion
  Model](https://arxiv.org/abs/2411.19108), 2024.
- Wei Fu et al.,
  [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for
  Language Reasoning](https://arxiv.org/abs/2505.24298), 2025.
- Shuo-yiin Chang et al.,
  [Joint Endpointing and Decoding with End-to-End
  Models](https://research.google/pubs/joint-endpointing-and-decoding-with-end-to-end-models/),
  2019.
- Loïc Barrault et al.,
  [Seamless: Multilingual Expressive and Streaming Speech
  Translation](https://ai.meta.com/research/publications/seamless-multilingual-expressive-and-streaming-speech-translation/),
  2023.

## Standards and operating references

- NVIDIA,
  [DCGM: Topology and NVLink](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/topology-and-links.html),
  for distinguishing topology inventory, link state, and measured traffic.
- MLCommons,
  [MLPerf Inference documentation](https://docs.mlcommons.org/inference/), for
  scenario, accuracy, and run-rule discipline.
- OpenTelemetry,
  [Semantic conventions](https://opentelemetry.io/docs/concepts/semantic-conventions/),
  for consistent trace, metric, log, and resource naming.
- NIST,
  [AI Risk Management Framework and Generative AI
  Profile](https://www.nist.gov/itl/ai-risk-management-framework).
- OWASP,
  [Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/).

## Official implementation documentation

- [vLLM documentation](https://docs.vllm.ai/en/stable/), including architecture,
  cache, parallelism, disaggregation, compilation, multimodal, and training
  integration guides.
- [vLLM architecture overview](https://docs.vllm.ai/en/stable/design/arch_overview.html).
- [SGLang documentation](https://docs.sglang.io/), including attention backends,
  distributed serving, observability, post-training, and diffusion.
- [SGLang Diffusion](https://docs.sglang.io/docs/sglang-diffusion).

Documentation can describe a release different from the pinned source snapshot.
When the two conflict, the manuscript either describes the pinned code or marks
the behavior as release-dependent.

## Publication tooling

Block diagrams are rendered with
[Mermaid 11.16.0](https://www.npmjs.com/package/mermaid/v/11.16.0), vendored
into the book itself (`assets/vendor/`), so diagrams render offline and are not
affected by CDN changes. The integration follows Mermaid's official
[initialization and rendering guidance](https://mermaid.js.org/config/usage.html).
Body, interface, and code typefaces (Literata, Inter, JetBrains Mono) are also
vendored as subsets. Diagram definitions remain readable as text if the
client-side renderer cannot load.

## Implementation map by chapter

| Chapters | vLLM paths | SGLang paths |
| --- | --- | --- |
| 1, 5 | `vllm/v1/engine`, `vllm/v1/executor`, `vllm/v1/worker` | `srt/managers`, `srt/model_executor` |
| 6 | `vllm/v1/core/sched/scheduler.py` | `srt/managers/scheduler.py`, `overlap_utils.py` |
| 7, 15 | `vllm/v1/core/kv_cache_manager.py`, `distributed/kv_transfer` | `srt/mem_cache/radix_cache.py`, `hiradix_cache.py` |
| 8 | `vllm/v1/attention/backends`, quantized and MoE kernels | `srt/layers/attention`, `kernels` |
| 9 | `vllm/compilation`, `vllm/v1/cudagraph_dispatcher.py` | `srt/model_executor/runner_backend`, `srt/compilation` |
| 10 | `model_executor/layers/quantization` | `srt/layers/quantization` |
| 11 | `vllm/v1/spec_decode`, `vllm/v1/structured_output` | `srt/speculative`, `srt/constrained` |
| 11b | `vllm/lora`, `vllm/v1/core/sched` (adapter-aware paths) | `srt/lora`, adapter manager paths |
| 12, 13 | `distributed/parallel_state.py`, `distributed/eplb` | `srt/distributed`, `srt/eplb` |
| 14 | `distributed/kv_transfer/kv_connector` | `srt/disaggregation` |
| 17 | scheduler encoder cache, `distributed/ec_transfer` | multimodal managers and encode disaggregation |
| 18 | diffusion model and runner paths | `multimodal_gen/runtime` |
| 19 | sleep and weight-transfer paths | scheduler and model-runner weight updaters |
| 21 | `entrypoints`, `parser`, `structured_output` | `srt/entrypoints`, `srt/constrained` |
| 22, 22b, 23 | benchmark and metrics packages, `/metrics` endpoint | benchmark, metrics, tracing, simulator, and `/get_server_info` |

## Reproducibility status

This manuscript explains how to design experiments but does not claim new
performance results. Numeric results cited from papers remain the authors'
results under their published setups. Future editions should attach original
benchmark cards, traces, commands, and raw data here, with each claim marked:

- **proposed** — experiment designed but not run;
- **reproduced** — run with public artifacts;
- **reviewed** — independently checked;
- **superseded** — retained for history but replaced by newer evidence.
