# Appendix A. Mathematical and Systems Notation

This appendix collects the quantities used throughout the book. The formulas
are estimates. Replace peak specifications with measurements when making a
capacity decision.

## Workload symbols

| Symbol | Meaning | Typical unit |
| --- | --- | --- |
| λ | request arrival rate | requests/second |
| `L_in` | input length | tokens, frames, or samples |
| `L_out` | output length | tokens, frames, or samples |
| `N` | active sequences or requests | count |
| `B_tok` | scheduled token budget per engine step | tokens |
| `W` | average time in the system | seconds |
| `Q` | average number of requests in the system | count |

For a stable system, Little's Law relates average concurrency, arrival rate,
and average time:

```text
Q = λ · W
```

The relationship is useful for checking measurements. If 20 requests arrive
per second and average end-to-end latency is 2 seconds, about 40 requests should
be in the system on average. It does not predict tail latency or guarantee
stability.

## Latency

For request `r`:

```text
TTFT(r) = time(first visible output) - time(arrival)

E2E(r)  = time(final output) - time(arrival)

TPOT(r) = (E2E(r) - TTFT(r)) / (L_out(r) - 1)
```

TPOT is defined only when more than one output token exists. Inter-token
latency is the sequence of gaps between visible output events. Report the
population, window, error treatment, and percentile method with every latency
distribution.

## Throughput and goodput

```text
request throughput = completed requests / duration

output throughput  = visible output tokens / duration

request goodput    = completed requests satisfying SLO / duration
```

Define whether cached input, padded work, rejected speculative tokens,
cancellations, and retries enter any numerator.

## Model memory

A simple parameter-memory estimate is:

```text
weight bytes = parameter count * average bits per parameter / 8
```

Include quantization scales, zero points, padding, embeddings, and replicated
parameters. A complete device budget is:

```text
device bytes = weights + persistent request state + activations
             + graph pools + communication buffers + allocator reserve
```

For a conventional KV layout:

```text
KV bytes per sequence
  = 2 * layers * tokens * KV heads * head dimension * bytes per element
```

The formula must be adapted for latent attention, sliding windows, recurrent
state, sharing, parallel sharding, and cache quantization.

## Compute and movement

Arithmetic intensity at one memory boundary is:

```text
I = operations / bytes moved
```

With peak compute `C` and bandwidth `B`:

```text
attainable operations per second <= min(C, B * I)
```

For a transfer of `S` bytes with startup latency `a` and sustained bandwidth
`b`, a first approximation is:

```text
transfer time = a + S / b
```

Concurrent transfers, contention, registration, serialization, and
synchronization add cost.

## Parallelism

Use these dimension names consistently:

| Abbreviation | Dimension |
| --- | --- |
| DP | data or request replicas |
| TP | tensor shards within layers |
| PP | pipeline stages across layers |
| EP | expert ownership shards |
| CP | context or sequence-position shards |
| SP | sequence parallelism; define the specific variant |
| DCP | decode-context parallelism |

Do not infer group composition from the product of sizes. Record the rank tuple
and communication groups explicitly.

## Cache value

Raw hit rate is:

```text
hit rate = cache lookups with any usable match / lookups
```

More useful measures include:

```text
token reuse rate = matched reusable tokens / eligible input tokens

saved compute per byte = estimated compute time avoided / bytes retained

transfer amplification = bytes moved through cache tiers / bytes consumed
```

Use measured prefill time where possible instead of assuming equal cost per
token.

## Cost

```text
cost per qualifying request = total service cost / SLO-qualified requests

cost per good output token = total service cost / output tokens from
                             SLO- and quality-qualified requests
```

State the accounting window and included infrastructure, software, labor,
network, storage, power, reservation, and failure costs.
