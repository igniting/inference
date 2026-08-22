# 2. Workloads, SLOs, and Goodput

Two teams benchmark the same model on the same GPU. One reports 20,000 output
tokens per second. The other reports that 95 percent of users see a first token
within 400 milliseconds. Which system is faster?

The numbers answer different questions. The first describes how much work the
server completed. The second describes how the service felt to most users.
Neither is sufficient on its own, and each can be made to flatter a failing
system: token counts rise when batches grow large enough to hurt latency, and
a latency percentile improves when slow requests are quietly excluded.

Before tuning an inference system, you need a precise description of the work
that arrives and the promises the service must keep. Otherwise, a benchmark can
improve while the product gets worse. This chapter builds the vocabulary that
makes such comparisons impossible to fake: units of work, the several latencies
hiding inside "how long did it take," percentiles and how to combine them,
capacity as distinct from throughput, goodput as the SLO-qualified rate, and
the workload description that all of it depends on.

## Visual map

**Goodput filters completed work through the product contract.**

```blockdiag
flowchart LR
    A["Arrivals"] --> B["Queue"]
    B --> C["Inference service"]
    C --> D["Completed requests"]
    D --> E{"Meets latency, quality, and correctness SLO?"}
    E -->|Yes| F["Goodput"]
    E -->|No| G["Completed but non-qualifying work"]
```

**The load generator changes what overload looks like.**

```blockdiag
flowchart TB
    O["Open-loop source"] -->|independent arrivals| S1["Server"]
    S1 --> Q["Queue can grow"]
    C["Closed-loop clients"] --> S2["Server"]
    S2 --> R["Responses"]
    R -->|permit next request| C
```

The first diagram is a filter, not a pipeline: completed work leaves the
service either way, and only the contract decides what counted. The second
diagram shows why the same server can produce two different overload stories.
An open-loop source keeps sending at its own rhythm while queues grow; a
closed-loop client cannot send its next request until the last one returns,
so rising latency silently throttles the load. Every measurement later in
this chapter is shaped by which of those two worlds produced it.

| Measure | Unit of observation | What it can hide |
| --- | --- | --- |
| TTFT | request | later stream stalls |
| ITL | token gap | initial queue and prefill |
| throughput | completed work per second | SLO failures and queue growth |
| goodput | qualifying work per second | reasons individual requests failed |

## Decide what counts as work

A text-generation service handles several nested units. A session contains
turns. A turn may produce one request. A request can create several candidate
sequences, and every sequence contains input and output tokens.

Media systems use different units: images, frames, audio chunks, latent patches,
or generated samples. Reinforcement-learning systems also group generations by
prompt and policy version.

This is why “requests per second” and “tokens per second” need qualifiers. A
request containing 50 input tokens is not the same job as one containing
50,000. Token throughput may count input tokens, output tokens, or both. It may
even include padded positions or speculative tokens that were later rejected —
work the hardware performed that no user received.

Retries create the same ambiguity at the request level. When a client times
out and sends the request again, the second attempt is new work to the server
but not to the user; a service that counts attempts reports higher volume
precisely when it is failing more often. The same accounting question
survives into goodput: if both attempts complete but only within-SLO attempts
qualify, the definition must say whether the retry's cost lands in the
denominator. Chapter 21 treats retry identity as an API contract for exactly
this reason — the metric story and the correctness story are the same story.

A useful metric always names its unit. For example:

> The service completed 320 successful requests per second, where requests had
> the production input and output length distribution.

That sentence is less impressive than a large unqualified number, but far more
useful. It is also worth treating metric definitions as part of the service's
contract, with the same discipline as any interface: written down, versioned,
and changed deliberately. When a dashboard redefines "latency" or a release
switches which tokens enter the numerator, every trend line built on the old
definition breaks without anything in the system failing. Chapters 21 and 22
return to this bookkeeping as an engineering obligation, not a reporting
courtesy.

Unit choice does not merely add precision — it can reverse a ranking. Assume
service A completes 100 requests per second at 500 output tokens each, and
service B completes 200 requests per second at 100 output tokens each. On
requests per second, B wins two to one. On output tokens per second, A wins
50,000 to 20,000. Both numbers are correct; they describe different products.
A caller chaining generations downstream cares about A's token rate, while a
caller issuing short classifications cares about B's request rate. Declaring
the unit is therefore part of declaring the audience.

## Latency is not one number

Consider a response that streams ten tokens. The first token appears after 600
milliseconds. Most later tokens arrive 40 milliseconds apart, but one gap lasts
half a second.

End-to-end latency tells you when the response finished. **Time to first token**
(TTFT) captures the initial wait. **Inter-token latency** (ITL) captures each gap
in the stream. **Time per output token** (TPOT) averages the time after the first
token across the remaining tokens:

```text
TPOT = (end-to-end latency - TTFT) / (output tokens - 1)
```

TPOT is compact, but it can hide the half-second pause. When output cadence
matters—as it does for chat, code completion, or speech—report ITL percentiles
or count stalls above a product threshold.

It also helps to break total latency into the stages a team can act on:

```text
network ingress
  + queueing
  + preprocessing
  + model execution and intermediate transfers
  + postprocessing
  + network egress
```

If a benchmark starts its timer after queueing and stops before streaming, it
does not measure the user's latency.

### Attributing a slow first token

The stage stack earns its place when it turns a complaint into an action.
Suppose users report that first tokens sometimes take over 600 ms, and traces
from one such request decompose the wait as follows: network ingress 10 ms,
edge queue 40 ms, engine queue 310 ms, preprocessing 15 ms, prefill 205 ms,
and egress 20 ms — about 600 ms in total.

Each stage names a different owner and a different remedy. The two queue
terms dominate at 350 ms combined, so no model or kernel work will fix this
request; the levers are admission policy, chunk sizing, and routing, which
are Chapters 6 and 16 subjects. Prefill's 205 ms is real computation, and
shortening it means prefix reuse or hardware, not scheduling. The remaining
45 ms of ingress, preprocessing, and egress is already near irreducible
floor. Attribution prevents the classic misresponse: tuning attention kernels
for a week because "the model felt slow," while the request spent half its
life waiting in a queue no dashboard displayed.

The decomposition also defines what a trace must record to be useful later.
A timestamp at each stage boundary costs microseconds at admission time and
is impossible to reconstruct afterward from end-to-end totals alone.

### Why a per-token p99 is not a per-request p99

Percentile claims inherit the population they are computed over, and token
gaps and requests are different populations. A small example makes the gap
impossible to ignore.

Suppose two requests are observed. Request A streams eleven tokens with gaps of
40 ms each. Request B streams two tokens, and its single inter-token gap lasts
500 ms. Pool all eleven gaps together and exactly one of them exceeds 400 ms:
the token-gap p99 sits near 500 ms, but the token-gap p90 is a comfortable
40 ms. Now count by request instead: one of the two requests contained a
half-second stall, so half of the users experienced it. No per-token
percentile below the extreme tail can express that.

TPOT averages the same evidence differently still. Request A contributes ten
40 ms gaps; request B contributes one 500 ms gap. The token-weighted average
is about 80 ms, because A's many well-behaved tokens outnumber B's single bad
one. The user-weighted story is that every second response stalled. None of
these numbers is wrong; each answers a different question. The failure mode
to avoid is quoting whichever one flatters the system, which is why the
reporting rules in the next section demand that the population be stated
every time.

## Why percentiles matter

An average combines ordinary requests with rare slow ones. In production, those
slow requests may be the exact cases a customer remembers.

The 99th percentile, or p99, is the value below which 99 percent of observations
fall in a stated population and time window. The population matters. A global
p99 can hide a small tenant that is consistently slow. A per-token p99 is not a
per-request p99. A number calculated from successful requests says nothing
about timeouts that were removed from the sample.

When reporting a percentile, state:

- what was observed: request, token gap, or session;
- which traffic was included;
- the test or production window; and
- how errors and cancellations were handled.

A well-formed claim reads like: "p99 of per-request TTFT, all tenants
including retries, measured at the edge over 09:00–10:00, cancellations
counted as failures." Every clause removes one way to misread the number,
and omitting any clause leaves the reader to guess — usually generously.

Do not average percentiles produced by separate hosts. Combine the underlying
samples or merge compatible histograms, then calculate the percentile.

### Merging histograms without lying

Production services report latency histograms per host, so combining them is
an everyday operation with two honest requirements: bucket boundaries must
match, and resolution must be reported honestly.

Assume two hosts expose request-latency histograms with boundaries at 100,
200, 400, and 800 ms. Host A served 100 requests with counts `[90, 8, 2, 0]`
per bucket; host B served 100 with counts `[95, 3, 1, 1]`. Merging is simple
addition per bucket: `[185, 11, 3, 1]` over 200 requests. The merged p50
falls in the 100–200 ms bucket, since the 100th ordered observation lands
there. For the merged p99, the 198th of 200 observations falls in the 400–800
ms bucket, so the honest statement is that p99 lies between 400 and 800 ms —
not a point value. If each host had exported finer buckets, the merged
estimate would tighten accordingly.

Both failure modes are now visible. Averaging the hosts' individual p99
values instead of merging would produce a number no user experienced. And
merging histograms with different boundaries silently fabricates precision:
the counts cannot be added because they describe different intervals. When
boundaries disagree, the only correct path is back to raw samples. Bucket
width bounds percentile resolution forever, which is a reason to choose
histogram layouts deliberately rather than accept a default.

The window deserves equal care. A p99 over a rolling five-minute window and a
p99 over the full day describe different services: the daily figure blends
the quiet night in and can hide an hour of morning degradation entirely,
while the short window surfaces it but also flatters any moment that happens
to follow a quiet stretch. Deployments add their own trap — a host that
joined mid-window contributes partial data unless its coverage is recorded.
State the window with the percentile, and when two windows disagree, treat
the shorter one as the more urgent message rather than the noisier one.

## Throughput is not capacity

Throughput measures completed work per unit time. Capacity is the arrival rate
the service can sustain while meeting its contract. The two diverge near
overload.

Imagine a server completing 100 requests per second while 120 arrive. Its
throughput looks stable, but the queue grows by 20 requests every second.
Latency will continue rising until callers time out or the system fails.

Appendix A collects a useful sanity check here. With λ the arrival
rate, `W` the average time in the system, and `Q` the average number of
requests present, Little's Law says `Q = λ · W`. In the overloaded
service, completion lags arrival by 20 requests per second, so `Q` climbs
without bound: after thirty seconds, roughly 600 requests are waiting whose
owners have not yet noticed. Running the law in the diagnostic direction is
just as valuable — measure any two of the three quantities and the third is
determined, so a dashboard showing stable `W` while λ rises must also
show `Q` rising somewhere, and if it does not, one of the measurements is
lying about its population. The law speaks in averages and assumes a stable
system; it predicts nothing about tails, which is precisely why percentiles
exist alongside it.

This leads to **goodput**: the rate of work that satisfies the service-level
objective, or SLO.

```text
request goodput = qualifying completed requests / test duration
```

A request might qualify only if it returns without error, begins within 500
milliseconds, maintains acceptable output cadence, and produces a valid result.
For a structured-output endpoint, malformed JSON does not count as goodput even
if it arrived quickly.

### Finding capacity by search, not assertion

Capacity is a property of the workload plus the contract, so it is found
empirically: offer increasing load and watch where qualifying work stops
keeping up. Assume an open-loop generator offers 60, 80, 100, and 120
requests per second against the same service, and the SLO-qualified
completion rates come back at 60, 79, 94, and 72.

| Offered rate | Qualifying rate | Reading |
| --- | --- | --- |
| 60 | 60 | every request qualifies |
| 80 | 79 | still keeping up |
| 100 | 94 | near the knee; queues forming |
| 120 | 72 | past the knee; goodput collapsing |

The service's capacity for this workload is roughly the offered rate where
goodput peaks — here near 100, where qualifying work is still rising but the
margin has vanished. Beyond the knee, extra offered load does not add
completed work; it displaces it, because arrivals spend longer in queues and
miss latency conditions they would have met at lower load. Note what the
experiment does not claim: the knee moves with the workload mix, the prefix
correlation, and the SLO clauses, which is exactly why "the service does 100
requests per second" is incomplete until its workload and contract travel
with it.

### Why the knee sits below 100 percent

The knee's position is not a policy choice; it falls out of utilization
arithmetic. Let ρ (Appendix A) be offered work divided by service capacity.
For random arrivals, the simplest queueing model — one shared queue,
exponentially spaced arrivals, exponential service — puts average waiting at
roughly `ρ / (1 − ρ)` service periods. The assumptions matter (real serving is
batched, correlated, and bimodal between prefill and decode), but the shape of
the curve survives every correction: waiting is proportional to ρ near zero
and diverges as ρ → 1.

Walk it in Atlas units. Suppose the knee experiment above found capacity near
100 requests per second, so one request occupies the system about 10 ms of
exclusive service time on average. At ρ = 0.5 the model predicts about `0.5 /
0.5 = 1` period of waiting; at ρ = 0.8 about four periods; at ρ = 0.9 about
nine. Going from half-used to 90-percent-used multiplies queue delay roughly
ninefold while raising throughput only 80 percent — and since TTFT includes
that delay, goodput collapses long before throughput does, which is exactly
what the 120-row in the table showed. This is why operating targets sit at
modest utilization: the last 20 percent of capacity costs more latency than
it returns work, and headroom is what absorbs arrival bursts. Any interviewer
asking "why not run hotter?" is really asking whether you can derive this
curve and name where its simplifications break.

Goodput often changes the winner in an architecture comparison. A large batch

The qualification clause is doing real work, so it deserves the same care as
the rate itself. Latency-qualified goodput, as above, is only one variant: a
structured-output endpoint qualifies on schema validity, a retrieval endpoint
on relevance thresholds, a media service on frame deadlines. Token-level
goodput — output tokens from qualifying requests — matters when callers
compose your service into longer pipelines, because downstream work scales
with tokens received, not requests observed. Whatever the clauses, they must
be written next to the number; "goodput" without its qualification is just
throughput wearing a better name.

### Closed-loop load is a thermostat

The closed-loop generator deserves arithmetic, because its self-throttling
behavior hides overload from unwary benchmarks. Consider a fixed population of
40 browser clients, each sending one request at a time. By Little's Law
rearranged, the offered rate is `Q / W` with `Q` fixed at 40: if a round trip
averages `W = 0.5` seconds, clients collectively offer 80 requests per second.

Now suppose the server degrades until `W` rises to 2 seconds. The client
population has not changed, yet offered load falls to 20 requests per second.
Queues drain, the server stabilizes, and measured latency settles at a value
that looks acceptable. The benchmark concludes the service survived; in
reality it collapsed to a quarter of its usefulness and the missing 60
requests per second are simply clients waiting for permission to speak. An
open-loop source offering 80 requests per second regardless of responses
would have exposed the same failure as unbounded queue growth within seconds.

Neither behavior is wrong — real browsers are closed-loop, so the effect is
physically real. The error is reading a closed-loop result as if it described
a capacity. State which loop produced a number, and treat closed-loop
"stability under overload" as the thermostat working, not the server coping.

## Describe the workload as a distribution

Suppose the support assistant receives mostly short questions. Ten percent of
users attach long documents, and half of all requests share one of a few system
prompts. Traffic is quiet overnight and arrives in bursts at the start of the
workday.

An average prompt length loses most of that information. A useful workload
record keeps the distributions of arrival time, input length, output length,
modality, media size, priority, tenant, and reusable prefix. It also preserves
correlations. Long documents may lead to long answers. Requests with the same
prefix may arrive together. Sampling each column independently creates a trace
that never existed — a synthetic morning where every long document gets a
short answer and prefix-sharing groups dissolve into unrelated traffic, so the
cache behaves in ways production never will.

Burstiness belongs in the record for the same reason. Total daily volume can
be identical between a flat day and a bursty one while peak queue depth
differs by an order of magnitude, and capacity decisions made against the
average will fail on the peak.

A workload record should also be replayable: timestamps, lengths, modality,
tenant, and prefix identifiers in a form a load generator can consume
directly. Replaying the same record against two engine revisions turns "the
new version feels faster" into a controlled comparison, and replaying a
recorded incident reproduces the arrival pattern that caused it. Chapter 22
builds its benchmarking discipline on exactly this foundation — without a
replayable description of the work, every performance claim is an anecdote.

Correlation is easiest to see in prefix sharing. Picture ten conversation
groups, each anchored by an 8,000-token system prompt. If requests from a
group arrive while its prefix is still resident, one prefill serves every
request that follows; if the same requests arrive scattered through the day,
the cache evicts between visits and each group's prefix is recomputed on
arrival. The input-token totals are identical — the correlated trace may
prefill 8,000 tokens where the dissolved one prefills hundreds of thousands.
Any capacity conclusion drawn from token totals alone is wrong in both
directions unless the arrival correlation traveled with the data.

Two load-generator styles answer different questions. An **open-loop**
generator sends work according to an external arrival process even when the
server slows. It exposes queue growth and overload. A **closed-loop** generator
waits for a response before sending the next request from a client. It models a
fixed client population, but as the thermostat example showed, it also reduces
offered load automatically when latency rises.

Neither style is universally correct. The mistake is failing to say which one
produced the result.

## Start from the product

Different products need different contracts.

An interactive assistant cares about TTFT, output cadence, cancellation, and
tail latency. An offline summarization job may accept high per-request latency
if a dataset finishes before a deadline. An embedding endpoint cares about
batch throughput and bounded completion time. A real-time media service has
frame or audio-chunk deadlines. A rollout service is coupled to a trainer and
may value policy freshness alongside generation speed.

Quality and correctness belong in every case. Quantization that meets the
latency target but damages an important task is not a successful optimization.
A tool call with the wrong schema is not useful output. A better service
balances several objectives rather than maximizing one:

```text
quality, correctness, availability, latency, goodput, cost, and energy
```

### Select the model with the system in view

Model selection is often presented as a leaderboard lookup. Inference
engineering turns it into a constrained product experiment.

Begin with an evaluation set drawn from actual product work: ordinary cases,
high-value cases, adversarial inputs, long contexts, required languages, tool
calls, and refusal behavior. Decide which failures are disqualifying before
comparing models — deciding afterward, once a favorite has emerged, converts
the evaluation into advocacy. The disqualifying set is product-specific: for
a coding agent, a tool call that writes to the wrong file is disqualifying no
matter how fluent the prose around it; for a brainstorming assistant, the
same mistake might be an ordinary quality ding. Then measure every viable candidate behind the serving stack
you could realistically operate.

A model with a higher offline quality score may be a worse product choice if it
misses the interaction deadline or requires a topology the team cannot keep
reliable. A smaller model may be preferable if retrieval supplies the missing
knowledge. Fine-tuning can change behavior without solving serving cost;
distillation can change both behavior and the execution envelope. Quantization
may let a candidate fit on fewer devices, but only an application evaluation
can determine whether its numerical changes are acceptable.

Write a one-page selection record for each serious candidate:

- the exact model, tokenizer, precision, context limit, and license;
- quality results on the product evaluation set;
- memory fit and required accelerator topology;
- latency and goodput under the target workload;
- operational dependencies and fallback behavior; and
- expected cost at ordinary and peak traffic.

The record prevents a common reversal: choosing a model in isolation and later
discovering that the service contract cannot afford it. Model and system
selection are one decision viewed at two levels.

## Worked example: throughput without goodput

One hundred requests arrive over 12.5 seconds. The server completes 96, so its
request throughput is 7.68 requests per second. Only 81 begin within 600 ms,
avoid token gaps above 150 ms, finish successfully, and return valid output.
Goodput is therefore 6.48 requests per second.

Walk the accounting. Completed work divides into three groups. Eighty-one
requests satisfy every clause of the SLO and count toward goodput. Fifteen
completed successfully but missed a latency condition — perhaps their TTFT
landed at 700 ms behind a burst of prefills. Four errored or timed out before
completing. Throughput counts the first two groups: 96 divided by 12.5 gives
7.68. Goodput counts only the first: 81 divided by 12.5 gives 6.48. The
fifteen-point spread between those rates is the entire content of this
chapter expressed as a number.

Removing the fifteen slow-but-completed requests from the latency sample would
make the percentile look better and destroy the meaning of the SLO. They must
remain completed work that failed to qualify. The four requests that errored or
timed out also remain in the workload accounting.

Now keep total tokens fixed but move arrivals into five bursts. Arithmetic work
is unchanged, yet queues form ahead of each burst and long prefills collide
with active decodes, so more requests miss the TTFT and ITL conditions and
goodput falls even though the server performs the same number of token
operations. The per-request traces make the mechanism visible: qualifying
failures cluster in the seconds after each burst begins, when queue depth
peaks. This is why the arrival process and length correlations belong in
the workload definition.

## Practice: construct comparable traces

Create three 100-request traces, each containing 100,000 input and 20,000 output
tokens: evenly spaced uniform requests, five bursts with mixed lengths, and ten
conversation groups sharing 8,000-token prefixes. Use an open-loop rate of 8
requests/s, then repeat closed-loop.

Report queue time, TTFT, worst per-request ITL, end-to-end latency, prefix
matches, errors, throughput, and the SLO-qualified goodput defined above.
Explain why equal token totals do not imply equal capacity. See the worked
construction in [Appendix G](../appendices/g-worked-solutions.md#2-workload-traces-and-goodput).

Now that the goals are clear, the next chapter looks inside the model to find
the computation and state that the server must schedule.
