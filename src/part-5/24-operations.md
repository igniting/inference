# 24. Observability, Reliability, and Operations

At 14:07, time to first token rises while GPU utilization falls. The model
workers report no errors. Is the cause a tokenizer backlog, a failed graph
capture, a cache-transfer timeout, a network partition, or an empty decode
pool?

Observability is the ability to answer that question from the system's outputs.
It begins with a model of the request path, not a large dashboard. Each
candidate cause lives at a different boundary — ingress, compilation, transfer,
membership, admission — so the system must expose a signal at every boundary
it owns, or the first incident becomes an archaeology dig with users as the
time pressure. "No errors reported" usually means "no error path was
instrumented," not "nothing is wrong."

The discipline mirrors Chapter 23's: a diagnostic is an experiment whose
question is "which component broke," and like any experiment it needs its
measurements designed before the event.

## Metrics show shape; traces show path

Metrics summarize behavior over time. Useful families include arrival and
completion rates, queue age, TTFT and ITL histograms, scheduled tokens, active
sequences, memory pressure, cache matches, transferred bytes, graph dispatch,
preemption, and errors.

**Operations needs signals from the request path and its resource owners.**

```blockdiag
flowchart LR
    R["Request path"] --> M["Metrics: rates and distributions"]
    R --> T["Traces: waits and boundaries"]
    R --> L["Logs: decisions and failures"]
    M --> D["Diagnosis"]
    T --> D
    L --> D
    D --> A["Safe action and rollback"]
```


Prefer queue *age* to queue *depth* as the headline signal — Chapter 21 made
the same discovery for realtime media. Depth conflates arrival rate with
service rate: forty queued requests during a healthy 45 ms-step regime is a
normal instant; forty queued requests whose oldest member has waited two
seconds is an incident in progress. Age is directly comparable against the
TTFT budget (how much of the 600 ms has the oldest request already spent?),
survives changes in batch composition, and degrades gracefully when request
sizes are heterogeneous. Record both if storage allows, but alert on age.

Logs record discrete decisions and failures. They should include request or
operation identity, component, state transition, version, and reason without
leaking prompt content or secrets.

Distributed traces follow one request across the router, preprocessing, engine,
stages, transfers, and output stream. A trace should distinguish waiting from
execution. The [OpenTelemetry semantic conventions](https://opentelemetry.io/docs/concepts/semantic-conventions/)
provide common naming principles for traces, metrics, logs, and resources,
including HTTP and RPC operations.

Use stable low-cardinality dimensions for metrics. Model, route, status, and
SLO class are often useful. Request ID, prompt hash, and tenant IDs belong in
traces or controlled logs; placing them in metric labels can overwhelm the
monitoring system and create privacy risk.

### Cardinality is a correctness constraint

Label choices have arithmetic consequences. Suppose a TTFT histogram with 40
buckets, crossed with route (5), SLO class (2), model (2), and — the tempting
mistake — tenant ID (say 500 active tenants). That is
`40 × 5 × 2 × 2 × 500 = 400,000` series per metric per replica; across eight
replicas and six such metrics, nearly twenty million active time series for
one signal family. Most tenants' series go stale between scrapes anyway, so
the store churns creating and expiring them — monitoring cost grows with
traffic mix rather than traffic volume, and queries that used to scan one
series per route now merge thousands. The privacy risk compounds it: tenant
IDs in labels leak who your customers are to everyone with dashboard access.

The working pattern: aggregate aggressively in metrics (per route and SLO
class, where capacity decisions live), and answer per-tenant questions from
traces or controlled logs where each record carries identity by construction.
If a per-tenant metric is genuinely required, make it an explicit allowlist of
large tenants, not an unbounded label.

Trace volume needs its own policy, and it is the mirror image of metric
cardinality: metrics must aggregate up front because they are unbounded over
time, while traces can be sampled because each one is individually complete.
Tail-based sampling keeps every trace that ended in error and every trace
slower than a threshold — precisely the ones diagnosis will ask for — and
samples the boring majority down to whatever storage supports. Head-based
sampling is cheaper and simpler but discards slow traces *because* they are
slow, hiding exactly the population the symptom table above routes on. Declare
the choice on the card; "we have tracing" without a retention statement
usually means the interesting traces were dropped first.

### Alert on symptoms, not causes

Alerts should fire on user-visible symptoms — SLO burn — and leave cause
hunting to humans with dashboards. Cause-based alerting ("cache hit rate below
60 percent," "queue depth above 50") pages on conditions that may not matter
today and misses the ones that do; symptom-based alerting needs only the SLOs
you already publish. The standard mechanism is multi-window burn rate: compare
error-budget consumption pace over a short window against a long one, and page
when both are elevated. Walked with Atlas numbers: a 99.9 percent availability
target leaves 0.1 percent of 30 days ≈ 43 minutes of error budget per month.
Paging when a one-hour window burns at 14 times the sustainable pace —
confirmed by a slower window so blips do not fire it — means that, if nothing
changed, the month's budget would be gone in `30 / 14.4 ≈ 2` days: early
enough to act, late enough that ordinary variation never wakes anyone. The
same structure applies to latency SLOs: TTFT-goodput is the availability
metric, its budget shrinks with every breaching request, and burn-rate paging
works unchanged. Every page should link the dashboard whose queries *are* the
first runbook branches; a page that does not start the diagnosis has wasted
its most expensive resource — a human's attention at 03:00.

## Observe the scheduler and state

GPU utilization alone cannot explain an inference engine. Record the waiting
and running request counts, oldest queue age, step token composition, prefill
chunks, decode batch size, admission rejection, and preemption.

For memory, record free and reserved blocks, allocation failure, fragmentation
or tail waste, live versus reusable state, and deferred release. For distributed
caches, include lookups, matched tokens, transfer duration, cancellation,
write-back, and stale location failures.

For MoE, record tokens per expert and per rank, dispatch and combine duration,
stragglers, and placement generation. For disaggregation, expose every stage
queue and transfer boundary. These metrics translate the architecture into
operational evidence: a Chapter 18 encoder queue age answers "is vision input
the bottleneck" in one glance; a Chapter 20 weight-version gauge confirms
every rank serves the same policy before you blame the model.

### Inside an engine's statistics layer

vLLM's
[`vllm/v1/metrics/stats.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/metrics/stats.py)
at the pinned SHA shows how much measurement philosophy fits in one dataclass
file. Cache hit rate is computed by `CachingMetrics` as a *sliding window over
the most recent requests* — a deque of `(requests, queries, hits)` trimmed to
a cap, defaulting to 1,000 requests — not a lifetime average, because a
lifetime average hides exactly the events operators care about: a cache flush,
a traffic-mix shift, an adapter rollout all move the *recent* rate long before
they dent the cumulative one. Its comment "DO NOT append empty stats to avoid
helpful info get[ting] kicked out" records a real bug class: empty updates
would dilute the window and silently drag the hit rate toward zero.

Eviction appears not as a counter but as *events*: `KVCacheEvictionEvent`
carries `lifetime_seconds`, `idle_seconds`, and a tuple of `reuse_gaps_seconds`
per block. That is eviction-policy evidence in recordable form — if blocks are
evicted idle-for-minutes and then requested seconds later, the retention
policy is wrong in a way no hit-rate scalar would localize. `SchedulerStats`
separates `prefix_cache_stats` from `connector_prefix_cache_stats`, keeping
local hits distinct from distributed-cache hits so a Chapter 15 connector
degradation cannot masquerade as a local-cache problem.

Two more details repay study. `RequestStateStats` keeps timestamps in two
domains on purpose: `arrival_time` is an "engine frontend timestamp
(wall-clock)," while `queued_ts`, `scheduled_ts`, `first_token_ts` are "engine
core timestamps (monotonic)" — the same dual-clock discipline Chapter 21
needed for media playout, applied to latency accounting, so cross-domain
subtractions are explicit rather than accidental. And `SchedulerIterationDetails`
carries an `is_dummy` flag alongside context-versus-generation token counts:
Chapter 14's participation steps surface as a metric field, letting a
dashboard separate real work from collective-synchronizing filler — the same
distinction Chapter 23 demands of any honest throughput claim.

## Readiness is a sequence of states

A process can be alive before it is ready. Model download, weight load,
distributed initialization, kernel compilation, graph capture, cache
registration, and router membership may all need to finish before traffic is
safe.

**Readiness progresses through model-specific startup stages.**

```blockdiag
flowchart LR
    P["Process alive"] --> W["Weights loaded"]
    W --> G["Distributed groups ready"]
    G --> C["Kernels compiled and graphs captured"]
    C --> H["Health execution passed"]
    H --> R["Router membership ready"]
```

| Symptom | First split | Evidence | Unsafe shortcut |
| --- | --- | --- | --- |
| high TTFT, low GPU use | ingress versus engine wait | queue ages and traces | add accelerators blindly |
| normal TTFT, high ITL | decode versus output path | step and stream gaps | tune prefill only |
| memory pressure | live versus reusable state | blocks, references, eviction | restart without leak check |
| one slow rank | compute versus communication | per-rank timeline | average utilization |


Liveness asks whether the process should be restarted. Readiness asks whether it
should receive new work. A worker draining old requests is live and not ready
for new ones. A worker blocked in a failed collective may have a running process
and be unable to make progress — Chapter 14's participation requirement means a
hung collective looks like a paused engine, not a crashed one, so neither probe
type catches it alone.

Health checks should test the dependency appropriate to their purpose. An HTTP
ping to the frontend does not prove the model group can execute. A full model
request can be too expensive for a frequent liveness probe.

Make the startup stages themselves observable: publish each transition in the
readiness diagram — weights loaded, groups joined, graphs captured, health
execution passed — as a timestamped event or gauge, so "the replica is stuck"
becomes "stuck at graph capture for six minutes," which names the component
and often the fix before anyone logs in. Chapter 9's explicit capture
signatures make this natural: the set of captured signatures *is* readiness
state, and reporting it costs one gauge.

### Choosing probes, and pricing their lies

| Probe | Proves | Cannot prove | Cost per call |
| --- | --- | --- | --- |
| TCP/port check | process bound the port | any model-path progress | negligible |
| HTTP liveness ping | frontend loop responsive | weights loaded, group joined | negligible |
| staged readiness gate | declared startup stage passed | current execution ability | none after startup |
| short health execution | one full forward pass works | tail shapes, all graphs | one engine step-ish |
| full canary request | end-to-end service behavior | nothing beyond its own shape | a real request |

Read the rightmost columns as the probe's blind spot. The dangerous failure is
not choosing the weak probe — it is asking a weak probe a strong question:
port checks answering "can this replica serve," or a single-shape health
execution answering "all captured graphs work." Chapter 9's explicit capture
signatures give the staged gate something concrete to report; a health
execution exercises one signature, so readiness should require the *set*, not
sample it. Price matters too: a full forward pass every five seconds steals a
batch slot from paying traffic on every replica — at Atlas's 45 ms step, a
health request landing each interval is a permanent tax of roughly one step in
a hundred at modest load. Frequent cheap probes plus infrequent expensive
ones beats one probe asked to do both.

## Treat the serving image as a measured artifact

"Same model" does not mean same service. A deployment is the combination of
weights, tokenizer, model code, engine revision, kernel libraries, accelerator
runtime, driver, configuration, and compiled artifacts. Pin and record that
combination as one release identity — Chapter 22 made template and parser
revisions part of served behavior, and operations extends the same identity to
everything below them.

Build containers from reproducible inputs and keep model artifacts outside the
mutable container layer when their size or access policy demands it. Verify
checksums before a worker becomes ready. Do not download unpinned executable
model code during startup. Produce a software bill of materials and scan both
the base image and Python or native dependencies, while recognizing that a
clean vulnerability scan does not prove model safety.

Startup time is operational capacity. Measure image pull, model fetch, weight
load, distributed initialization, compilation, graph capture, and warm-up
individually. If a worker takes twelve minutes to become ready, an autoscaler
cannot rescue a two-minute traffic spike. Warm pools, local artifact caches, or
forecast scaling may be required — and the warm-pool size is arithmetic, not
vibes: if demand can double within two minutes and a replacement worker needs
twelve, the pool must already hold enough ready workers to absorb the entire
spike, because *none* of the reactive capacity arrives in time. Twelve-minute
startup converts elasticity from a control loop into a procurement decision.

### Scale to zero, honestly priced

The logical extreme of elasticity is serving nothing when traffic is absent
and paying only for what runs — attractive for bursty internal tools and
multi-tenant platforms, and it lives or dies on the readiness sequence
above. What can actually be made cheap? Weights are the bulk: pre-staged on
local disk and warmed into the page cache, they load in seconds rather than
minutes — Chapter 20's sleep levels already priced the extreme at a ~3 s host
snapshot against a multi-minute cold envelope. Compilation is next: captured
graphs and tuned kernels serialize as artifacts (Chapter 9) if their capture
was deterministic in shape set and environment, restoring in seconds;
recapture costs minutes. Distributed setup and health execution take
seconds more. The floor for a large model is therefore tens of seconds to
low minutes even with everything staged — dominated by weight streaming and
collective bring-up — and *nothing about the KV cache survives*, because
session state cannot be snapshotted into an artifact.

That last clause sets the product contract. Scale-to-zero serves cold-start
tolerant traffic: batch jobs, scheduled workloads, tenants whose first
request may be slow but whose tenth is warm. Interactive traffic needs the
warm-pool arithmetic above instead, and the honest comparison is per tier:
pool cost per hour versus lost-or-delayed requests during ramp. An
interviewer probing "why doesn't everyone scale to zero?" wants exactly this
split — what stages compress to seconds, which one dominates what remains,
and why the state that makes inference *good* is precisely the state that
cannot ride in the artifact.

Promote the same immutable artifact through staging and production. Environment
configuration may change endpoints and capacity, but rebuilding between stages
removes much of the evidence gathered by the canary. Store the release identity
on every trace so an output or latency regression can be tied back to the exact
execution environment.

## Overload should fail deliberately

When queues exceed the service's ability to recover within the SLO, reject or
shed work before the deployment collapses. Preserve capacity for health,
cancellation, and high-priority traffic. The rejection itself must be priced
like Chapter 17's admission veto: early and loud beats late and silent, and a
shed request returns an actionable class (limit, overload, deadline) so callers
can respond intelligently instead of retrying into the collapse.

Graceful modes may reduce maximum output, disable expensive optional features,
route to a smaller model, lower media quality, or pause background work. Each
mode needs a product and correctness contract — Chapter 21's degradation ladder
is the realtime instance, but batch endpoints deserve the same pre-agreed
answers to "what may we stop doing."

An error budget connects reliability targets to change velocity. Track failures
caused by overload separately from model validation, dependency failure, and
internal bugs. They need different remedies: overload wants admission and
capacity work, validation wants gates in the release pipeline, dependency
failure wants isolation and fallbacks — and spending the budget on the wrong
category buys nothing.

## Test failures on purpose

Kill one worker in a tensor-parallel group. Partition a cache from its metadata
service. Delay a KV transfer. Exhaust host memory. Return a late completion
after cancellation. Corrupt a downloaded model artifact in a staging
environment.

For every test, observe detection time, user impact, cleanup, retry behavior,
and recovery. A failover that restores traffic while leaking blocks will cause
a second incident later — the pass criterion includes the Chapter 22 assertion
that KV references return to baseline, not merely that errors stopped.

Detection time has an architecture-implied bound worth checking in the drill.
If router membership uses a lease with a 5-second renewal period, a dead
tensor-parallel rank should leave routing within roughly one lease period
after its group stalls — Chapter 17's membership epochs are the fence. User
impact should then be bounded by drain behavior: reusing Chapter 20's handoff
arithmetic, draining 400 running requests at one 45 ms engine step each takes
about 18 seconds, so a clean failover of a loaded replica costs roughly that
long of elevated latency on the affected route. If the drill shows 4 minutes
of errors instead of 5 seconds of detection plus 18 seconds of drain, the gap
*is* the finding — something between the lease and the router is not
propagating membership, and no amount of capacity would have fixed it.

Disaggregated systems deserve coupled tests. If the decode pool fails while
prefill remains healthy, admission should stop before completed KV state piles
up. If the remote cache fails, the service may degrade to recomputation rather
than becoming unavailable — at Chapter 17's prices, recomputing a cached 4,000-
token prefix costs about `4,000 × 0.06 = 240 ms` of rework, which is the
number that decides whether cache-loss degradation meets the TTFT budget or
should shed load instead.

## Deploy without mixing incompatible state

A rolling deployment needs a model and engine compatibility boundary. Drain
requests before replacing workers that own nonmigratable state. Keep cache and
artifact namespaces separate across versions. Do not send a live session to a
new tokenizer or weight version without an explicit migration — Chapter 20's
weight transactions and Chapter 17's cache-version bumping are the mechanisms;
deployment orchestration is what must refuse to bypass them.

Canary traffic should represent the shapes and features most likely to expose
problems: long context, structured output, multimodal input, adapters, and
distributed modes. Compare output correctness and goodput, not only error rate.
Because every trace carries the release identity (the artifact section's
requirement), canary evaluation becomes a join rather than an inference:
group traces by release, compare TTFT-goodput and conformance results per
group, and a regression names its version instead of a time window that two
changes share.

Rollback must remain possible after caches, schemas, or control-plane metadata
change. Test it before the incident.

## Write runbooks around hypotheses

A useful runbook starts from a symptom and branches on evidence.

For high TTFT with low GPU utilization, check ingress and tokenizer queues,
prefill admission, cache-transfer waits, graph warm-up, and worker readiness. For
high ITL with normal TTFT, inspect mixed prefill chunks, decode batch size,
collective stragglers, output processing, and session transport.

Each step should name the metric or trace, expected range, safe action, and
rollback. Avoid instructions that say "restart the service" without identifying
which state will be lost. A branch whose action destroys the evidence that
selected it — restarting workers before recording queue ages and admission
reasons — converts an explainable incident into folklore.

Reversibility is the quiet requirement behind every safe action: each runbook
step should state not only its rollback but *when* to exercise it ("remove the
temporary capacity after the admission queue drains," not "eventually"). An
irreversible action can still be correct — failing over a dead replica
destroys its local cache state by design — but then the runbook owes the
reader an explicit list of what was destroyed and what must re-warm, which is
exactly what Appendix G's drill dashboard measures. Post-incident review
closes the loop: every real incident should end by adding or correcting one
branch, one expected range, or one missing metric, so the runbook converges on
the system's actual failure modes rather than its imagined ones.

## Worked example: high TTFT, low GPU use

p95 TTFT rises from 480 ms to 1.4 seconds while GPU utilization falls from 72
to 38 percent. Take the two numbers together before acting: falling
utilization means accelerators are starving, not saturating, so adding
capacity treats a symptom and hides the cause. The Appendix G runbook branches
on evidence, each step naming its confirming signal.

First split: ingress and tokenizer queue age. If those queues are deep, the
engine is innocent — route around or scale that tier, and rollback is simply
removing the temporary capacity once the backlog drains. If they are normal,
inspect engine admission age and *reasons*: a surge in remote-cache waits
points at a Chapter 15 dependency, not insufficient GPUs. Third, compare
scheduled prefill tokens against graph-fallback and compilation events — a new
shape compiling mid-incident argues for stopping canary traffic on that route,
retaining the old artifact, rather than touching workers at all. Only then do
readiness and collective health enter, and a failed group leaves routing
*before* any restart, preserving live state where possible.

Suppose the cause is the failure drill's 500 ms KV-transfer delay. Correct
behavior is bounded transfer waiting, then conditional recomputation — about
240 ms per affected 4,000-token prefix, per the arithmetic above — or early
rejection when even that breaches the remaining TTFT budget. Restarting
workers first destroys the cached state *and* the evidence while leaving the
dependency failure untouched. The dashboard that makes this diagnosable shows
stage queue ages, transfer counts and bytes, timeout reasons, recomputation
counts, and end-to-end goodput — every field mapping to one branch above.

## Practice: write and test the runbook

Build a dashboard for the Chapter 15 pipeline and inject 500 ms into KV
transfers. Write the high-TTFT/low-utilization runbook with expected metric
ranges, safe actions, and rollback at every branch.

Measure detection, user impact, cancellation cleanup, recomputation, leaked
blocks, and recovery. Give the runbook to an engineer who did not build the
system. The worked branch structure is in
[Appendix G](../appendices/g-worked-solutions.md#24-operations-runbook).

The final chapter brings the technical choices together with cost, security,
and organizational ownership.
