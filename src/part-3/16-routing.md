# 16. Routing, Replication, and the Control Plane

Once a service has several replicas, the frontend must decide where each
request goes. Round-robin routing is attractive because it needs little state.
It is also blind to the two resources that dominate inference: queued work and
warm model state.

Routing is the cluster-level version of scheduling. The local scheduler chooses
the next work on one engine. The router chooses which engine should own a new
request.

## A replica has more state than “healthy”

Useful routing information can include queue length, estimated remaining work,
free KV blocks, active batch composition, cached prefixes, adapters, model
version, current stage role, and recent failures.

No router sees all of this perfectly. Telemetry arrives late. A decision based
on an empty queue may reach the worker after several other requests. A cache
entry can be evicted between lookup and assignment.

Treat routing data as a prediction. Once a worker accepts a request, it should
become the authority for that request's local lifecycle. The global router
should not micromanage every token step.

## Common routing policies

Round robin spreads request counts. Least-connections spreads active requests.
Least-estimated-work tries to include prompt and expected output length.
Session affinity keeps related turns together. Cache-aware routing values saved
prefill. Priority-aware routing reserves capacity for important traffic.

Each policy sees only part of the cost. A good practical score can combine
estimated queue time, execution work, cache savings, transfer cost, and a
penalty for uncertain or stale telemetry.

The weights should come from measurement. A cached token has little value if
the worker's queue is several seconds long. An idle replica is less attractive
if it lacks a required adapter and must load it first.

[Preble](https://arxiv.org/abs/2407.00023) studies this conflict directly: a
distributed prompt scheduler must co-optimize reusable prefix state and load,
because maximizing either one alone can make placement worse.

## Sessions need a state policy

Affinity is useful for multi-turn chat and real-time media because the worker
already holds state. It also makes a worker failure or hotspot more disruptive.

Decide whether session state can migrate, be reconstructed, or is lost with the
worker. A short chat prefix may be cheap to recompute. A long video session with
recurrent state may need replication or checkpointing. The routing policy
follows from that state policy.

Sticky routing should have an escape. If the preferred worker is overloaded or
draining, the router can transfer state, recompute on another replica, or reject
according to the remaining deadline.

## Global admission and backpressure

A fleet can be overloaded even when some workers still accept requests. The
control plane needs a view of total queued work and stage capacity.

In a prefill/decode deployment, admission should consider both pools. Sending a
request into an available prefill worker is harmful if no decode capacity will
be ready afterward. In an MoE deployment, a network or expert hotspot can limit
capacity while aggregate GPU utilization looks low.

Global admission can reserve capacity, reject work that cannot meet its SLO, or
return a retry delay. Backpressure should reach the original caller or durable
upstream queue. Uncoordinated retries multiply load precisely when the service
has the least spare capacity.

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

## Membership and deployment

When a replica joins, the router must not send traffic until weights, parallel
groups, graphs, and health checks are ready. When it leaves, new traffic should
stop before current work drains. Forced termination needs a retry and state-loss
policy.

Rolling out a new model version creates two cache namespaces and possibly two
sets of compiled artifacts. Requests in a session should not cross versions
accidentally. Canary routing must compare equivalent traffic and keep the old
version available for rollback.

Membership changes are distributed events. Use generations or epochs so a
delayed health message from an old process cannot make a dead replica current
again.

## Failure changes routing cost

A timeout can mean a slow request, a failed worker, a partitioned network, or an
overloaded dependency. Retrying on another replica may recover and duplicate
expensive work. Hedging can reduce tail latency while consuming extra capacity.

Requests should carry stable IDs and attempts. Output protocols need a rule for
which attempt is authoritative. State transfers and cache writes should be
idempotent or safely abandoned. The router should open a circuit around a
failing destination rather than continuing to discover the same failure per
request.

## Simulate locality under load

Create a router simulator with several replicas, skewed prefix popularity, and
bursty arrivals. Give each replica a local scheduler and finite cache. Compare
round robin, least estimated work, cache-only routing, and a hybrid cost model.

Plot goodput, queue percentiles, recomputed tokens, cache occupancy, and traffic
imbalance. Increase the popularity of one prefix until cache-only routing loses
to recomputation on idle replicas. Then allow replication of the hot prefix and
measure the capacity it consumes.

The simulation completes the path from one request to a distributed text
service. Part IV applies the same ideas to models whose serving loops are not
limited to text decode.
