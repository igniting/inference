# Appendix C. Reproducible Benchmark Cookbook

This appendix provides a compact format for the experiments used in the book.
It is intentionally engine-neutral.

## Benchmark card

Create one card per result.

```yaml
decision: "Does configuration A improve TTFT/ITL-qualified goodput?"
hypothesis: "Smaller prefill chunks reduce decode stalls at a throughput cost."
date: "YYYY-MM-DD"

model:
  identifier: "MODEL_ID"
  revision: "MODEL_REVISION"
  tokenizer_revision: "TOKENIZER_REVISION"
  precision: "bf16"
  quantization: null

software:
  engine: "ENGINE_NAME"
  commit: "GIT_SHA"
  container_digest: "sha256:..."
  driver: "DRIVER_VERSION"
  accelerator_runtime: "RUNTIME_VERSION"

hardware:
  accelerator: "DEVICE_MODEL"
  count: 8
  cpu: "CPU_MODEL"
  host_memory_gib: 1024
  topology_file: "artifacts/topology.txt"

execution:
  tensor_parallel: 8
  pipeline_parallel: 1
  data_parallel: 1
  expert_parallel: 1
  graph_mode: "describe exact mode"
  attention_backend: "BACKEND"

workload:
  trace: "traces/workload.jsonl"
  arrival: "open-loop Poisson, 8 requests/second"
  warmup_requests: 200
  measured_requests: 5000
  cache_state: "empty at warm-up start; retained during measurement"
  seed: 1234

slo:
  ttft_p99_ms: 800
  itl_p99_ms: 80
  max_error_rate: 0.001
  quality_gate: "EVAL_NAME >= VALUE"

artifacts:
  command: "commands/run.sh"
  raw_results: "results/raw.jsonl"
  summary: "results/summary.json"
  traces: "results/traces/"
```

YAML is used here for readability. Store exact commands and raw data as files,
not only in prose.

## Workload trace schema

One JSON object per request is easy to stream and inspect:

```json
{
  "request_id": "trace-000001",
  "arrival_ms": 0,
  "input_token_ids": [101, 202, 303],
  "max_output_tokens": 128,
  "priority": 0,
  "tenant_class": "interactive",
  "session_id": null,
  "media": [],
  "sampling": {"temperature": 0.0},
  "expected_schema": null
}
```

Production-derived traces should remove content and identifiers according to
policy while preserving length, timing, prefix-sharing, and correlation needed
by the experiment.

## Experiment sequence

1. Verify model output and protocol behavior on a small golden set.
2. Record the environment and topology.
3. Run cold-start measurement if it is part of the decision.
4. Warm the intended compilation and cache paths.
5. Confirm that no unexpected compilation or fallback continues.
6. Run the workload at several offered-load points.
7. Capture detailed profiles only at representative regimes.
8. Repeat and retain every raw result, including failures.
9. Run quality and semantic-equivalence checks.
10. Write a conditional conclusion and its falsification boundary.

## Required plots

For online generation, prefer:

- offered load versus SLO-qualified goodput;
- TTFT and ITL percentile curves versus load;
- queue time by stage;
- active batch and scheduled token distributions;
- cache matched tokens and transfer bytes;
- error, cancellation, and preemption rates;
- resource utilization by stage and rank;
- cost per qualifying request.

Do not truncate axes in a way that exaggerates small differences. Show
uncertainty or repeated runs. Label cold, warm, and steady-state regions.

## Fair comparison checklist

- Same model weights, tokenizer, template, precision, and quality target
- Same hardware allocation, power state, and topology
- Same request trace and arrival behavior
- Same context and output limits
- Same streaming and stop semantics
- Same cache starting condition
- Equivalent warm-up and compilation treatment
- Tuning policy described for every system
- Errors and timeouts retained in the denominator
- Raw commands and engine-specific configuration published

If equivalent semantics cannot be achieved, report the difference and avoid a
single winner label.

## Result statement template

```text
Under [workload] on [hardware], using [model and revisions], configuration A
changed [primary metric] from X to Y while satisfying [quality and SLO gates].
The observed mechanism was [evidence from timeline/counters]. The result did
not hold under [boundary condition]. Raw artifacts are at [path].
```
