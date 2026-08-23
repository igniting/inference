# 17. Routing, Replication, and the Control Plane

Once a service has several replicas, the frontend must decide where each
request goes. Round-robin routing is attractive because it needs little state.
It is also blind to the two resources that dominate inference: queued work and
warm model state. The blindness is measurable with the book's own constants.
Two identical 1,000-token prompts arrive together; round robin sends one to an
idle replica, where TTFT is `20 + 0.035 × 1,000 ≈ 55 ms` plus one decode step,
and the other behind a queued 40,000-token prompt whose prefill alone costs
`20 + 0.035 × 40,000 ≈ 1,420 ms`. Counts stayed balanced; time-to-first-token
differed by roughly 1.4 seconds. A router that cannot see either queues or
state distributes requests evenly and serves them unevenly — identical counts
arrive at each replica while completion times diverge by seconds.

Routing is the cluster-level version of scheduling. The local scheduler chooses
the next work on one engine. The router chooses which engine should own a new
request. The same principles apply at both levels — estimate service time,
admit against capacity, protect SLOs — but the router works with worse
information: its view of every replica is a report about the past.

## A replica has more state than “healthy”

Useful routing information can include queue length, estimated remaining work,
free KV blocks, active batch composition, cached prefixes, adapters, model
version, current stage role, and recent failures. Each field earns its place
on the report only if some policy consumes it, and different policies lean on
almost disjoint subsets:

**The global router predicts a destination; the local scheduler owns execution.**
Telemetry flows one way, decisions the other, and the loop closes only as fast
as reports travel.

```blockdiag
flowchart LR
    A["Request and deadline"] --> R["Global router"]
    T["Stale queue and locality telemetry"] --> R
    R --> W1["Replica 1 scheduler"]
    R --> W2["Replica 2 scheduler"]
    R --> W3["Replica 3 scheduler"]
    W1 --> T
    W2 --> T
    W3 --> T
```


| Reported field | Consumed by | Failure when absent |
| --- | --- | --- |
| queue depth / est. remaining | least-work, hybrid queue term | hot-spotting by count |
| free KV blocks | admission, decode-side placement | accepted requests swap or preempt |
| cached-prefix inventory | cache-aware term, affinity | systematic recompute |
| adapters held, model version | score load/risk terms | surprise cold loads, version skew |
| stage role (prefill/decode) | coupled-pool admission | decode starvation behind full prefills |

No router sees all of this perfectly. Telemetry arrives late. A decision based
on an empty queue may reach the worker after several other requests. A cache
entry can be evicted between lookup and assignment.

Treat routing data as a prediction. Once a worker accepts a request, it should
become the authority for that request's local lifecycle. The global router
should not micromanage every token step.

### The telemetry budget

Staleness has a price in requests, and Appendix A's Little's law prices it:
at arrival rate λ and reporting delay d, the router's picture of any queue is
wrong by roughly λ·d requests. Assume a ten-replica fleet taking 40 requests
per second in total — 4 per second per replica — and telemetry published every
250 ms. Each router decision is made against a snapshot that is, on average,
one request out of date per replica; during a burst arriving at twice the
average, two. That is the error floor under *normal* operation, before any
network partition or slow heartbeat stretches d. It explains an empirical
rule: routing improvements from better policies shrink, and can reverse, when
telemetry intervals stretch past a few hundred milliseconds, because the
prediction error begins to exceed the differences between destinations.

Age also determines *what kind* of decision an observation can still support,
which is why serious routers tag every report with its timestamp and let the
policy — not the data path — decide how much to trust it:

| Observation age | Still supports | No longer supports |
| --- | --- | --- |
| ≤ 50 ms | ordering near-identical queues | trusting exact cache contents |
| ≤ 250 ms | policy selection, coarse load spreading | queue-position claims |
| ≤ 1 s | capacity class: admit, shed, drain | any ordering claim |
| unknown | treat as the oldest case | everything above |

The budget cuts both ways. Publishing telemetry more often costs control-plane
bandwidth and scheduler time — every report interrupts the engine step loop it
describes — and finer-grained data goes staler faster because it describes a
smaller, faster-changing quantity. Queue depth in requests is stable enough to
report at 250 ms; free KV blocks change every admission and finish; cached-
prefix inventories are large, so most deployments advertise them as compact
sketches or on-change summaries rather than full listings. Match the report
cadence to how fast each quantity actually moves.

## Common routing policies

Round robin spreads request counts. Least-connections spreads active requests.
Least-estimated-work tries to include prompt and expected output length.
Session affinity keeps related turns together. Cache-aware routing values saved
prefill. Priority-aware routing reserves capacity for important traffic.

**A hybrid routing score compares waiting, recomputation, and movement.**

```blockdiag
flowchart TB
    C["Candidate replica"] --> Q["Estimate queue time"]
    C --> P["Estimate missing-prefix compute"]
    C --> T["Estimate transfer or adapter load"]
    Q --> S["Combined cost plus uncertainty"]
    P --> S
    T --> S
    S --> D["Choose destination and record prediction"]
```

| Policy | Sees queue? | Sees locality? | Characteristic failure |
| --- | --- | --- | --- |
| Round robin | no | no | unequal work per request |
| Least work | yes | no | repeated expensive prefill |
| Cache only | no | yes | hot cached replica |
| Hybrid cost | estimated | estimated | stale or misweighted predictions |


Each policy sees only part of the cost. A good practical score can combine
estimated queue time, execution work, cache savings, transfer cost, and a
penalty for uncertain or stale telemetry.

The weights should come from measurement. A cached token has little value if
the worker's queue is several seconds long. An idle replica is less attractive
if it lacks a required adapter and must load it first.

[Preble](https://arxiv.org/abs/2407.00023) studies this conflict directly: a
distributed prompt scheduler must co-optimize reusable prefix state and load,
because maximizing either one alone can make placement worse.

### Scoring a placement

A hybrid score is just the worked example's arithmetic generalized, term by
term:

```text
cost(R) = queue(R) + missing_tokens(R) × 0.06 ms
        + transfer_or_load(R) + risk(R)
```

The first two terms produce G §16's table: R0 at 300 + 0, R1 at 0 + 240,
R2 at 100 + 120. The third term catches what the simple score ignores. Give
R1 a required adapter it does not hold: assume loading that adapter costs
800 ms of foreground time on first use. R1's cost becomes 0 + 240 + 800 =
1,040 ms and it drops from second place to last — a policy that ignored the
adapter term would have sent every subsequent adapter-sharing request there
too, paying 800 ms each time until the adapter warmed. Transfer works the
same way in disaggregated deployments: Chapter 15 priced a 6,000-token KV
move at ~95 ms, which is exactly the kind of term that belongs here rather
than being discovered after admission.

The risk term prices the telemetry budget from the previous dive: a
destination whose last report is old, or whose locality claim comes from a
sketch rather than a confirmed pin, carries a penalty proportional to how
wrong it could be. In practice the penalty needs to be asymmetric. Overestimating
a busy replica's cost sends traffic somewhere slightly worse; underestimating
it stacks another request onto a queue that was already the bottleneck. Set
the asymmetry from measurement of your own staleness distribution, not from
symmetry aesthetics.

### Pricing a cached prefix

Cache-aware routing needs a price for locality, and the pinned constants give
one for the worked example's 4,000-token prefix. Its KV image weighs 4,000 ×
320 KiB ≈ 1.22 GiB, so three ways of honoring a match to it differ sharply:

| Strategy | Cost per hit | Fleet-level bill at 10 hits/s | State held |
| --- | --- | --- | --- |
| recompute locally | 240 ms of prefill (0.06 ms × 4,000) | 2.4 engine-seconds of prefill per second | none |
| fetch from a shared copy | 12 ms setup + 1.22 GiB ÷ 22 GiB/s ≈ 68 ms | 12.2 GiB/s of link traffic toward one holder | one 1.22 GiB copy |
| replicate to every replica | ≈ 0 ms | negligible | 1.22 GiB × replicas of HBM |

None dominates. Recompute burns prefill-engine time that other requests
wanted; the shared copy converts a compute problem into a network problem —
10 hits per second pull 12.2 GiB/s through the holder's links, over half of
one NVLink-class link, and make that replica both popular and fragile, which
is exactly G's hot-cache hazard; replication spends HBM that otherwise holds
about two-thirds of another 6,000-token conversation. The crossover variable
is hit rate. At 0.1 hits per second the recompute bill is 24 ms of engine
time per second — noise, and no memory or link is worth spending on it. At
10 hits per second replication buys back a full engine's worth of prefill for
a few GiB. Hit rate, however, drifts: a prefix becomes popular with a news
cycle and fades with it. Replication decisions therefore need the same
hysteresis as autoscaling — promote on sustained hit rate, demote on sustained
absence — because a policy that flips on every crossing thrashes memory and
invalidation traffic alike. Chapter 16's invalidation ladder prices the
demotion side of that flip.

## Sessions need a state policy

Affinity is useful for multi-turn chat and real-time media because the worker
already holds state. It also makes a worker failure or hotspot more disruptive.

Decide whether session state can migrate, be reconstructed, or is lost with
the worker. A short chat prefix may be cheap to recompute. A long video session with
recurrent state may need replication or checkpointing. The routing policy
follows from that state policy:

| State kind | Recovery on failover | Affinity strength |
| --- | --- | --- |
| short text prefix | recompute (tens of ms) | soft — escape freely |
| long document prefix | recompute (seconds) or distributed-cache hit | medium — prefer, then escape |
| media/recurrent state | checkpoint or replicate | hard — drain before removal |

Sticky routing should have an escape. If the preferred worker is overloaded or
draining, the router can transfer state, recompute on another replica, or reject
according to the remaining deadline. The escape condition should be evaluated
per turn, not per session — a sticky session that cannot break affinity turns
its worker's queue into the user's latency.

## Global admission and backpressure

A fleet can be overloaded even when some workers still accept requests. The
control plane needs a view of total queued work and stage capacity.

In a prefill/decode deployment, admission should consider both pools. Sending a
request into an available prefill worker is harmful if no decode capacity will
be ready afterward — Chapter 15's coupled queues are the mechanism, and the
router sits close enough to see both sides. In an MoE deployment, a network or expert hotspot can limit
capacity while aggregate GPU utilization looks low; Chapter 14's balancedness
statistic is the sort of signal that distinguishes "spare capacity" from
"capacity gated by one rank."

Global admission can reserve capacity, reject work that cannot meet its SLO, or
return a retry delay. Backpressure should reach the original caller or durable
upstream queue. Uncoordinated retries multiply load precisely when the service
has the least spare capacity — a retry storm converts a 30-second overload
into a five-minute one, because every timed-out client re-submits exactly when
workers are least able to help.

### Admission is routing with a veto

The hybrid score ranks destinations; admission decides whether the request may
go anywhere, and the Atlas SLO turns that veto into arithmetic. Suppose the
request arrives at the router having already spent 350 ms in an upstream
gateway. TTFT must stay within 600 ms, so the remaining budget is 250 ms.
Score the same three replicas against it: R1 survives with 10 ms of margin
(its cost is 240 ms) and R2 with 30 ms (220 ms), while R0's 300 ms makes it
inadmissible outright. Tighten the upstream spend to
400 ms and the budget falls to 200 ms: *no* destination qualifies. The correct
action flips from routing to rejecting — return a retry delay or shed — before
any worker sees the request.

That flip is worth internalizing because admitting into certain violation is
not neutral. An admitted request that misses its deadline still consumes its
prefill and decode slots, pushing neighbors' queues longer and converting one
missed SLO into several. Early rejection spends nothing but the caller's
patience, and the retry-delay estimate comes from the same score: if R2's cost
is 220 ms now, a delay of a few hundred milliseconds plausibly restores
admissibility, whereas a blind immediate retry lands on unchanged queues. The
veto also protects the deadline term from being gamed implicitly — a score
without admission quietly routes deadline-starved requests to whichever
replica misses by the least.

## Autoscaling has memory

Traditional autoscaling often reacts to CPU utilization or request count. An
inference replica has long startup stages: image pull, model load, distributed
initialization, compilation, graph capture, and cache warm-up. By the time a new
replica is ready, the original burst may be over.

Useful signals include queueing delay, SLO headroom, estimated work, KV pressure,
stage imbalance, and sustained arrival trends. Scaling policy needs hysteresis
so the fleet does not repeatedly add and remove replicas around one threshold.

Scale-down also costs state. Draining a warm replica can discard valuable
prefixes or sessions. Compare the saved capacity cost with the future cold
penalty. A minimum warm pool may be cheaper than scale-to-zero for latency-
sensitive models.

### Startup latency versus reaction time

Assume the pinned models' startup sequence lands at four minutes end to end
— image pull tens of seconds, weight load dominating, then compilation and
graph capture from Chapters 8–9 adding minutes before the first useful step,
then cache warm-up. A reactive autoscaler that triggers when utilization
crosses a threshold for two consecutive windows adds capacity four minutes
after the signal. Any burst shorter than that pays nothing and costs a
replica; any burst longer gets relief only for its tail. This is why
inference scaling signals lead rather than lag: queueing delay and arrival
trend predict the need roughly one startup-duration ahead, and the scale-out
decision is really a forecast with a four-minute horizon.

Hysteresis sizing follows from the same number. The down-scale threshold
should sit far enough below the up-scale threshold that normal traffic noise
cannot traverse both within one drain cycle, and drained replicas should
release state in the order Chapter 16's invalidation ladder defines —
visibility first, storage later — so a flapping fleet does not thrash the
distributed cache along with itself.

## Membership and deployment

When a replica joins, the router must not send traffic until weights, parallel
groups, graphs, and health checks are ready. When it leaves, new traffic should
stop before current work drains. Forced termination needs a retry and state-loss
policy.

Rolling out a new model version creates two cache namespaces and possibly two
sets of compiled artifacts. Requests in a session should not cross versions
accidentally. Canary routing must compare equivalent traffic and keep the old
version available for rollback.

The namespaces interact with locality in a way that is easy to miss at rollout
time. A hot prefix replicated across the fleet before the rollout — per the
pricing dive above — is warm only in v1's namespace; a canary replica holding
v2 weights cannot read those blocks, so every canary request pays full recompute
and looks artificially slow against v1 traffic that rides warm caches. The
comparison is not measuring model quality; it is measuring cache temperature.
Fair canaries either exclude cached-prefix advantages from both sides, route
each version its own warm-up period, or compare only on prompts known to be
cold for both namespaces.

Membership changes are distributed events. Use generations or epochs so a
delayed health message from an old process cannot make a dead replica current
again. The failure this prevents has a specific shape: replica X crashes, its
replacement registers, then a queued health report from X arrives and a router
that keys on content rather than epoch marks both alive and splits traffic to
a process that no longer exists. An epoch number on every membership message
makes the stale report self-invalidating.

## Failure changes routing cost

A timeout can mean a slow request, a failed worker, a partitioned network, or an
overloaded dependency. Retrying on another replica may recover and duplicate
expensive work. Hedging can reduce tail latency while consuming extra capacity.

Requests should carry stable IDs and attempts. Output protocols need a rule for
which attempt is authoritative. State transfers and cache writes should be
idempotent or safely abandoned. The router should open a circuit around a
failing destination rather than continuing to discover the same failure per
request:

| Circuit state | Behavior | Transition out |
| --- | --- | --- |
| closed | route normally; count recent failures | failures exceed threshold → open |
| open | exclude destination entirely | probe timer expires → half-open |
| half-open | admit a single probe request | probe succeeds → closed; fails → open |

The half-open probe matters for inference specifically because recovery is
not instantaneous: a replica that just restarted must reload weights, rebuild
graphs, and warm caches before it serves honestly. A probe that succeeds at
health-endpoint level while the first real request still pays cold-start costs
will flap the circuit. Gate the transition back to closed on the same readiness
signals the membership protocol used at join time.

Hedging has a computable break-even even without a distribution handy. Assume
1 percent of requests strand on a slow path costing 3 extra seconds. Hedging
all of them after 500 ms spends one duplicated request — including its prefill,
which Chapter 4 prices from prompt length — per hundred requests, to remove up
to 2.5 seconds from 1 percent of tails. Whether that trade is good depends on
what a duplicated prefill displaces at the second replica, which is why hedge
decisions belong in the same scoring framework as everything else in this
chapter rather than in a static percentage.

## Worked example: partial locality wins

R0 has all 4,000 reusable tokens and 300 ms of queue. R1 is idle with no match.
R2 has 2,000 matched tokens and 100 ms of queue. If recomputation costs 0.06 ms
per missing token, estimated placement costs are 300, 240, and 220 ms. R2 wins.

Walk each term. R0: perfect locality, zero prefill, but the request inherits
every queued request ahead of it — 300 ms of other people's work. R1: zero
queue, but 4,000 missing tokens at 0.06 ms each = 240 ms of prefill. R2:
2,000 matched tokens leave 2,000 to compute = 120 ms, behind 100 ms of queue;
100 + 120 = 220 ms, twenty milliseconds under R1 and eighty under R0. The
margin is thin — which is the point. Shift R2's queue to 150 ms and R1 wins;
shift R0's queue to 250 ms and R0 wins. Partial-locality routing earns its
complexity only because these quantities move on the scale of single
requests, and the score re-evaluates per arrival.

Cache-only routing chooses R0; least-queue chooses R1. A hybrid cost makes its
assumptions visible and can add transfer, adapter load, stale-telemetry risk,
and deadline penalty. It remains a prediction, so the worker reports the actual
queue and match observed at admission — closing the loop and giving the next
revision of weights its training data.

## Practice: simulate a hot prefix

Implement the three-replica case, then add bursty arrivals, delayed telemetry,
finite caches, and one increasingly popular prefix. Compare round robin,
least-work, cache-only, and the hybrid estimate.

Plot goodput, queue percentiles, recomputed tokens, occupancy, and imbalance.
Find when replication of the hot prefix repays its memory and add hysteresis to
prevent oscillation. Delay telemetry deliberately in the simulator — G's note
is worth keeping: perfect instantaneous queue knowledge would make the router
unrealistically powerful. The worked decision is in
[Appendix G](../appendices/g-worked-solutions.md#17-cache-aware-routing).

The simulation completes the path from one request to a distributed text
service. Part IV applies the same ideas to models whose serving loops are not
limited to text decode.
