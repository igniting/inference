# 26. Security, Isolation, and Governance

An inference service is a high-throughput interpreter for untrusted prompts,
media, schemas, model artifacts, and tool results. Its performance
mechanisms—shared caches, reusable adapters, remote connectors, compiled graphs,
and administrative APIs—also create trust boundaries. Security therefore
belongs in the architecture of the request path, not in a deployment checklist
added afterward.

This chapter follows data and authority through that path. It asks what one
tenant can infer about another, which inputs execute code or allocate scarce
resources, which artifacts join the trusted computing base, and how deletion
reaches derived state that has spread across a fleet.

## Multi-tenancy needs isolation at every layer

Tenants share queues, model weights, memory allocators, caches, network links,
and sometimes adapters. Quotas should cover concurrent requests, token work,
media processing, cache occupancy, and expensive features—not only request
rate. The Chapter 22 limit-pricing arithmetic is the template: a quota that
counts requests treats a 500-token summarization and a 12,000-token analysis
with media as equals, which they are not — token-work quotas approximate the
real scarce quantities, KV blocks and engine-step time.

**Shared compute does not imply shared identity or cache state.**

```blockdiag
flowchart LR
    A["Tenant A"] --> QA["Identity and quota"]
    B["Tenant B"] --> QB["Identity and quota"]
    QA --> S["Shared scheduler and model"]
    QB --> S
    S --> CA["Namespace A cache"]
    S --> CB["Namespace B cache"]
    S --> P["Explicit public namespace"]
```

Memory must be cleared or safely overwritten before reuse across trust
boundaries. Cache lookup needs namespaces and authorization. Metrics and logs
must not expose prompts, token IDs, or identifying hashes. Timing can also leak
whether another tenant has warmed a prefix.

Noisy-neighbor controls need a defined unit. Equal requests are not fair when
one tenant submits 100-token prompts and another submits 100,000-token
prompts: under request-count quotas, ten equal requests let the second tenant
consume roughly a thousand times the KV blocks (`100,000 × 320 KiB` against
`100 × 320 KiB` per request) and orders of magnitude more engine-step time.
Token-work quotas collapse that asymmetry by construction, and estimated-time
quotas — predicted cost from Chapter 17's scoring, charged before admission
and reconciled after — handle modalities where token counts understate real
work, like decoded media or expensive sampling modes. No unit is final;
the requirement is that the unit track something the shared infrastructure
actually competes for.

### What one tenant's warm prefix tells another

Cross-tenant cache hits are a correctness question disguised as an
optimization. Suppose tenants A and B both submit prompts sharing a 4,000-token
document prefix. Without namespaces, B's request hits A's warmed blocks and
skips roughly 68 ms of fetch plus the matching prefill share — and the *timing
delta itself* discloses information: B can probe whether some document has
been served recently by measuring whether its TTFT drops. That is a real
channel even when no bytes ever cross tenants, and it is why Atlas defaults
tenant cache namespaces to isolated: the shared-document case must opt in
through an explicit sharing policy, carrying authorization with it.

Isolation has a price, and quoting it keeps the default honest. With
namespaces, B recomputes the prefix — at Chapter 17's recompute price about
240 ms of engine work for 4,000 tokens — so aggregate compute rises in
proportion to how much cross-tenant overlap existed. Measure that overlap
before assuming it matters; teams that enable sharing "for efficiency"
without measuring routinely discover the overlap was a handful of system
prompts, which can be handled with a public, non-secret shared namespace
instead of weakening tenant boundaries everywhere.

## The model server executes untrusted inputs

Prompts can attempt to manipulate application behavior. Media can exploit
parsers. Structured schemas can consume compiler resources. Tool output can
contain instructions aimed at the model. Generated code or tool calls can
affect external systems if the application grants authority.

Each row deserves its own containment, because they fail differently. Media
parsers run native code on attacker-controlled bytes — Chapter 18's decode-
before-admit pipeline means hostile images execute *inside* your service
boundary, so parser choice, sandboxing, and decoded-media limits (Chapter
21's) carry security weight beyond latency. Grammar complexity is the schema
analogue of a decompression bomb: Chapter 22 priced compilation off the hot
path precisely so a pathological schema costs a Future, not an engine step —
but unbounded grammar size still consumes memory and compile threads, so
complexity limits belong in the quota list. Tool output round-trips through
the model as new input, closing a loop where an external page can instruct
the model as fluently as its user can.

Treat model output as untrusted data. Validate it at the action boundary, apply
least privilege, require confirmation for high-impact actions, and use
idempotency for retries — Chapter 22's separate tool-execution idempotency key
is here because a replayed model response must not become a repeated wire
transfer. Prompt-based defenses do not replace access control.

## Availability attacks look like difficult workloads

An inference service can be denied without exploiting memory corruption. An
attacker can submit inputs that are valid but expensive: maximum-length
prompts, decompression-heavy media, grammars with pathological compile cost,
outputs that never reach a stop condition, low-reuse adapter churn, or prefixes
chosen to pollute a shared cache. These requests pass a simple schema check and
consume the same scarce resources as valuable work.

| Input pattern | Scarce resource | Limit before expensive work | Runtime control |
| --- | --- | --- | --- |
| long prompt or output | KV blocks and engine steps | token and context ceiling | per-tenant token-work budget and deadline |
| compressed or oversized media | CPU, decoded memory, encoder time | byte, dimension, frame, and decoded-size limits | sandboxed decode and media-work quota |
| complex schema | compiler CPU and parser state | grammar size and construct limits | bounded compile pool, timeout, and cache quota |
| adapter churn | host bandwidth and accelerator memory | approved adapter identity and size | load-rate and residency quota |
| prefix-cache pollution | cache occupancy and metadata | authenticated namespace and key size | tenant occupancy budget and admission value |
| recursive tool loop | external authority and wall time | tool allow-list and turn budget | action count, idempotency, timeout, and fence |

Enforce the cheapest limits first. Authentication and compressed-byte limits
belong before media decode; token and grammar estimates belong before engine
admission; tool authority belongs at the action boundary. A limit applied only
after allocation protects the response contract but not capacity. Chapter 2's
overload invariant becomes a security invariant here: once admitted work
exceeds bounded service capacity, honest users experience the attack as an SLO
failure.

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
software bill of materials. Chapter 24's release identity is the runtime half
of this: pinning is only useful if the deployment *refuses* unpinned inputs,
which is what checksum-before-ready enforces.

The contract surfaces from Chapter 22 belong in the trusted base too.
Tokenizers, chat templates, tool parsers, and grammar backends are executable
artifacts that shape every response; a tampered template can redirect a
conversation as effectively as tampered weights, and neither the benchmark
suite nor the conformance suite catches it unless both are re-run when those
artifacts change — which is exactly why Chapter 18 folded processor version
into execution identity and why release identity pins all of them together.
Adapter artifacts get the same treatment as model code: they are weights
contributed by someone, loaded into shared memory beside the base model, so
they inherit the review, pinning, and scanning requirements rather than
slipping in through a side door.

Separate public inference credentials from management APIs that load models,
update weights, inspect memory, run profilers, or execute collective RPCs. These
capabilities can alter outputs, extract information, or deny service — Chapter
19 made weight updates transactional, and Chapter 22 refused to route them
through the public network; supply-chain hygiene is the same boundary viewed
from the build side. The drill list from Chapter 24 belongs here as well: a
staging exercise that swaps a signed artifact for an unsigned one should end
in a readiness refusal, not a warning in a log nobody reads.

## Retention applies to derived state

A deletion policy that removes prompts but leaves KV blocks, encoder features,
logs, traces, or benchmark samples is incomplete. Derived state can preserve
information about the original input.

Map every data class through the request lifecycle:

| Data class | Derived from | Outlives the request as |
| --- | --- | --- |
| KV blocks | prompt and generated tokens | reusable state until eviction or deletion |
| encoder features | media input | cached embeddings keyed by content hash |
| traces and spans | whole request path | diagnosis records with timing and metadata |
| logs | decisions and failures | operation history, hopefully prompt-free |
| benchmark samples | recorded traffic | evaluation fixtures with long lifetimes |

Define retention, encryption, region, access, and deletion for each tier. Ensure
backups and distributed caches honor the same model. A cache's performance
value does not override a user's deletion right or contractual boundary — which
is why Chapter 17's cache-version bumping and namespace machinery matter beyond
routing: deletion across a distributed cache is an invalidation sweep with an
audit trail, not an `rm`.

The sweep has to reach every replica the block migrated to — Chapter 15's
transfer machinery moves KV state between nodes precisely so it survives, and
survival is the problem here — plus encoder-feature caches keyed by content
hash, where deleting the *key mapping* must also age out the stored features.
Backups restore old state wholesale, so their retention model bounds every
tier's effective deletion date: a cache that honors deletion instantly but sits
on a filesystem backed up for ninety days has a ninety-day deletion policy
whether anyone chose it or not.

## Design the trust boundaries

Draw the public request path, the model and adapter supply path, and the
administrative path separately. Mark where identities change, where untrusted
bytes become executable work, where state crosses tenant or region boundaries,
and where an action receives external authority. A boundary is incomplete until
it names authentication, authorization, limits, audit evidence, and failure
behavior.

**Public, supply, management, and action paths cross different boundaries.**

```blockdiag
flowchart TB
    C["Public clients"] -->|Untrusted prompts, media, schemas| I["Inference boundary"]
    A["Artifact registry"] -->|Signed models and adapters| S["Supply boundary"]
    O["Operators"] -->|Privileged control| M["Management boundary"]
    I --> W["Model workers"]
    S --> W
    M --> W
    W -->|Proposed calls| X["Action boundary"]
    X --> E["External systems"]
```

Keep public generation credentials separate from credentials that load models,
attach storage, update weights, inspect memory, invoke profilers, or execute
distributed control operations. A server that exposes both paths under one
authority has made every prompt-facing parser part of its management plane.

## Governance turns boundaries into evidence

Governance is the machinery that keeps a reviewed boundary from drifting. Keep
an inventory that joins every production route to its model, tokenizer,
template, parser, adapter policy, runtime image, data regions, owner, and
approval record. That is the security view of Chapter 24's release identity.
A model alias without a digest or an endpoint without an owner is an unmanaged
change surface.

Review is triggered by changed authority or data flow, not by model size alone.
A new tool, remote-code model, shared cache namespace, telemetry field, region,
or provider can change the threat model without changing a single kernel.
Conversely, a kernel-only upgrade may use the existing review when its signed
artifact, conformance, isolation, and rollback evidence all remain inside the
approved boundary. Record exceptions with an owner and expiry; a permanent
"temporary" bypass is an undocumented architecture decision.

Security incidents need inference-specific evidence: request and tenant IDs,
admission and quota decisions, artifact digests, cache namespaces and
invalidation generations, administrative operations, tool proposals and
confirmations, and deletion acknowledgements. Preserve that evidence without
logging raw prompts by default. Drill at least the failures the design claims
to contain—cross-tenant cache probes, unsigned artifact loads, runaway schemas,
stale tool results, and deletion across a restored backup—then make the failed
control observable in the same dashboard that operators already use.

## Worked example: a cache hit becomes a side channel

Two tenants submit a shared 4,000-token document prefix. A namespace-free cache
lets the second request skip prefill, creating a measurable TTFT difference even
though no cache bytes are returned. The safer default isolates tenant
namespaces. A deliberately public corpus may use a separate shared namespace,
but its authorization and retention policy travel with the cache key.

The performance cost is recomputation; the security benefit is that latency no
longer reveals another tenant's recent work. Measure the overlap before
weakening the boundary. If only a few public system prompts are shared, publish
those explicitly instead of enabling cross-tenant reuse globally.

## Practice: produce a threat model and deletion proof

For the Atlas deployment, enumerate trust zones for clients, routing, model
workers, caches, object storage, model artifacts, adapters, tools, and
administrative APIs. For each crossing, state the identity, allowed action,
resource limit, retained evidence, and fail-closed behavior.

Then delete one request containing text and media. Trace every derivative—tokens,
KV blocks, encoder features, logs, traces, benchmark samples, distributed-cache
replicas, and backups—and provide evidence that each tier either removed the
state or expired it under a declared retention bound. Compare with the worked
solution in [Appendix G](../appendices/g-worked-solutions.md#26-security-boundaries).

A secure inference architecture does not promise that untrusted input becomes
safe because a model processed it. It limits what each identity can spend,
observe, retain, and cause. The appendices that follow collect the notation,
hardware and portability reference, reproducibility templates, deployment and
decision checklists, terminology, source provenance, worked solutions,
migration guide, and debugging playbook used throughout the book.
