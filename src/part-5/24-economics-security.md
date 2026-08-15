# 24. Economics, Security, and Architecture Decisions

The fastest configuration is not always the one a team should deploy. It may
require scarce hardware, duplicate too many weights, expose administrative
interfaces, or cost more per useful answer. Architecture is the process of
making those constraints explicit.

## Choose an economic unit that reflects value

Cost per GPU-hour is an input, not an outcome. Cost per request ignores sequence
length. Cost per output token ignores quality, retries, and latency failures.

A stronger unit is cost per qualifying request or cost per good output token.
It includes only work that meets the quality and service contract introduced in
Chapter 2.

```text
unit cost = total serving cost / qualifying work
```

Total cost includes accelerators, CPUs, host memory, storage, network transfer,
reserved but idle capacity, software and operations, and failed or repeated
work. For owned hardware, include depreciation, power, cooling, support, and
the cost of capacity that cannot be reassigned.

Compare steady and bursty workloads. A design with excellent saturated
efficiency may be expensive at the product's normal utilization.

## Utilization can hide stranded resources

An MoE deployment may show high network use and low expert compute. A
disaggregated service may have a full decode pool and idle prefill GPUs. A model
can consume nearly all HBM while leaving arithmetic units underused.

Report utilization by resource and stage. The limiting resource determines
capacity; the others may be stranded. Independent scaling helps only when pool
sizes can track the workload without adding excessive transfer or warm-up cost.

Power limits can change kernel clocks and throughput. Energy per useful output
captures a dimension that device-hour pricing may hide. If carbon-aware
scheduling is a requirement, deadlines and data locality constrain when and
where offline work can move.

## Multi-tenancy needs isolation at every layer

Tenants share queues, model weights, memory allocators, caches, network links,
and sometimes adapters. Quotas should cover concurrent requests, token work,
media processing, cache occupancy, and expensive features—not only request
rate.

Memory must be cleared or safely overwritten before reuse across trust
boundaries. Cache lookup needs namespaces and authorization. Metrics and logs
must not expose prompts, token IDs, or identifying hashes. Timing can also leak
whether another tenant has warmed a prefix.

Noisy-neighbor controls need a defined unit. Equal requests are not fair when
one tenant submits 100-token prompts and another submits 100,000-token prompts.
Token or estimated-time quotas are better starting points, with corrections for
modalities and model paths.

## The model server executes untrusted inputs

Prompts can attempt to manipulate application behavior. Media can exploit
parsers. Structured schemas can consume compiler resources. Tool output can
contain instructions aimed at the model. Generated code or tool calls can
affect external systems if the application grants authority.

Treat model output as untrusted data. Validate it at the action boundary, apply
least privilege, require confirmation for high-impact actions, and use
idempotency for retries. Prompt-based defenses do not replace access control.

The [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)
provides a maintained taxonomy that includes prompt injection, sensitive
information disclosure, supply-chain risk, improper output handling, excessive
agency, and resource consumption. The
[NIST Generative AI Profile](https://www.nist.gov/itl/ai-risk-management-framework)
places these technical risks within a broader process for governing,
measuring, and managing AI risk.

## Protect the inference supply chain

Model repositories can include executable custom code, serialized objects,
tokenizers, templates, and native kernels. Engine plugins and JIT compilation
expand the trusted computing base.

Pin model and container digests. Verify signatures or checksums. Prefer safe
serialization formats. Review custom model code before enabling it. Build
kernels and images in controlled environments, scan dependencies, and retain a
software bill of materials.

Separate public inference credentials from management APIs that load models,
update weights, inspect memory, run profilers, or execute collective RPCs. These
capabilities can alter outputs, extract information, or deny service.

## Retention applies to derived state

A deletion policy that removes prompts but leaves KV blocks, encoder features,
logs, traces, or benchmark samples is incomplete. Derived state can preserve
information about the original input.

Map every data class through the request lifecycle. Define retention,
encryption, region, access, and deletion for each tier. Ensure backups and
distributed caches honor the same model. A cache's performance value does not
override a user's deletion right or contractual boundary.

## Managed service, self-hosted, or hybrid

A managed API transfers responsibility for engine operation and capacity while
limiting control over weights, placement, and low-level optimization.
Self-hosting provides control and creates responsibility for security,
reliability, upgrades, and hardware supply. Hybrid designs may use managed
capacity for bursts or selected models.

Compare options using the same service contract. Include engineering and
on-call cost, time to support new models, compliance, portability, failure
independence, and exit cost. A lower accelerator rate can be more expensive if
the team cannot keep the deployment reliable.

## Write the architecture decision

An architecture decision record for an inference service should contain:

- workload distributions and growth assumptions;
- quality, latency, availability, and cost targets;
- model stages and persistent state;
- hardware and network topology;
- parallel, scheduling, cache, and routing plans;
- overload, failure, deployment, and rollback behavior;
- security boundaries and data retention;
- benchmark evidence and rejected alternatives;
- assumptions that trigger a future review.

The rejected alternatives matter. They show which constraints led to the
decision and prevent a future team from repeating the same investigation
without new evidence.

## Worked example: a decision with triggers

Atlas begins with self-managed four-way tensor-parallel replicas, continuous
batching, local prefix caching, and hybrid queue-plus-locality routing. Prefill
and decode remain colocated until measured long-prompt interference repays the
KV transfer boundary. A managed API is an explicit overflow route, not an
invisible retry.

TP8 is rejected because wider layer-frequency collectives hurt the interactive
regime. Unconditional disaggregation is rejected because short prompts do not
repay transfer. Tenant caches default to isolated, and model artifacts,
administrative controls, and public generation use separate security
boundaries.

The decision reopens if context length makes KV capacity binding, prefix reuse
falls below its routing benefit, bursts become shorter than worker startup, or
the TTFT objective tightens enough to justify separate prefill capacity.

## Practice: write the capstone ADR

Produce the Atlas architecture record using the workload and dense model from
Chapters 2–4. Include topology, scheduling, caching, routing, overload,
deployment, rollback, data retention, benchmark evidence, and rejected plans.

Change traffic, context, prefix reuse, hardware price, and SLO one at a time.
For each, name the threshold that triggers review rather than merely stating
that cost changes. The worked ADR is in
[Appendix G](../appendices/g-worked-solutions.md#24-architecture-decision).

A sound architecture is not the answer to one benchmark. It is a decision whose
assumptions and failure modes are visible.

That completes the main text. The appendices provide notation, reference
tables, reproducibility templates, deployment patterns, terminology, and the
source ledger behind this edition.
