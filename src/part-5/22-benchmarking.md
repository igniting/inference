# 22. Benchmarking and Performance Science

A benchmark is an experiment designed to answer a decision. Without the
decision, it becomes a number generator.

"Which engine is fastest?" is too broad. A useful question is narrower:

> For our model, hardware, request distribution, and TTFT/ITL targets, does
> prefix-aware routing increase goodput enough to justify its control-plane
> complexity?

That question tells you what to hold constant, what to vary, and which result
matters. It also tells you what the benchmark is allowed to *cost*: a question
that gates a routing rewrite justifies a week of careful measurement; a
question that gates a kernel flag does not. Budget the experiment like any
other engineering artifact.

The failure mode this chapter exists to prevent is the confident wrong
number: a result that is precisely measured, cleanly plotted, and answers a
question nobody asked — or worse, answers it under conditions so different
from production that the decision it drives is a coin flip with extra steps.

## Visual map

**A benchmark is an evidence loop, not a single load-generator run.**

```blockdiag
flowchart LR
    H["Falsifiable hypothesis"] --> W["Representative workload"]
    W --> R["Controlled repeated runs"]
    R --> A["Raw events and analysis"]
    A --> Q["Quality and SLO gate"]
    Q --> C["Conditional conclusion"]
    C --> H
```

**Offered load must be swept through the service's operating regimes.**

```blockdiag
flowchart LR
    L["Low load"] --> S["Saturation approach"]
    S --> O["Overload"]
    L --> M1["Latency floor"]
    S --> M2["Goodput knee"]
    O --> M3["Queue growth and rejection"]
```

| Benchmark layer | Controlled input | Required output | Frequent mistake |
| --- | --- | --- | --- |
| Microbenchmark | operation and shape | latency and numerical error | claiming service speedup |
| Engine | batch and state | step time and resource trace | excluding preparation |
| Service | arrivals and requests | latency, errors, throughput, goodput | closed-loop overload masking |
| Product | task population | usefulness and cost | optimizing invalid output |

## Begin with a hypothesis

Write the expected causal chain before running the test. For example:

```text
smaller prefill chunks
  -> shorter mixed engine steps
  -> lower decode stalls
  -> better ITL goodput
  -> possibly lower prefill efficiency and worse TTFT
```

This prediction determines the measurements. If you record only total tokens
per second, you cannot test it. The chain also names its own confounders: the
last line predicts a *cost*, so the experiment must measure TTFT even though
the hypothesis is about ITL — a benchmark that only instruments its hoped-for
effect is an advertisement, not an experiment.

Also write a falsification condition. If smaller chunks lower ITL but reduce
goodput after TTFT constraints are included, the proposed configuration did not
achieve its goal. Decide the threshold now — "goodput must improve by at least
5 percent or we keep the simpler configuration" — because a threshold chosen
after seeing the data will always be met by something. Pin the analysis
revision with the hypothesis too: which script, over which raw events,
computes the verdict. A result whose analysis can drift after collection is
not yet an experiment.

## Use the right benchmark level

A kernel microbenchmark isolates one operation and is ideal for comparing
implementations across shapes. An engine-step benchmark includes input
preparation, kernels, graphs, and collectives. An end-to-end load test includes
queues, routing, preprocessing, streaming, and clients.

Each level removes noise and context. Start at the lowest level that can answer
the question, then verify at the service boundary. A kernel speedup that
vanishes at the engine step is still useful diagnostic evidence; it is not an
end-to-end performance result.

### The engine-step level, walked

The engine level earns its keep because step composition is where most
scheduler claims live, and it is cheap to instrument. Using Atlas's frozen
costs: a pure-decode step at the operating batch runs 45 ms. A mixed step that
admits a 256-token prefill chunk pays roughly `20 + 0.035 × 256 ≈ 29 ms` of
prefill work (the frozen prefill model, applied to a chunk), so the mixed step
costs on the order of 45 ms of decode work *plus* that chunk — decode tokens in
that step see their ITL stretch by however much the chunk extends the step.
That arithmetic is the whole hypothesis of chunked prefill in miniature: the
chunk size dial trades prefill throughput against a per-step ITL tax that the
service-level benchmark will later either forgive (ITL budget 150 ms has room)
or punish (tail ITL already near budget). Running the sweep at the engine level
first costs minutes; discovering the tax at the service level costs a full
load-test cycle.

What the engine level cannot tell you is equally important: queueing,
admission, routing, and client behavior are absent by construction. An engine
result is a *component* input to the service decision, never the decision.

### Microbenchmarks and their traps

Below the engine level sit kernel microbenchmarks, and they have failure modes
of their own worth naming because vLLM's pinned tree dedicates whole directories
to getting them right —
[`benchmarks/kernels`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/benchmarks/kernels)
and `fused_kernels`, plus an `overheads` directory for framework-level fixed
costs. Four traps account for most wrong microbenchmark numbers:

- **Warm caches measure a different kernel.** Repeating one shape back-to-back
  keeps weights and activations resident in levels of the memory hierarchy that
  a serving step — interleaved with other layers' traffic — never enjoys.
  Rotate shapes or flush deliberately.
- **Clock resolution versus kernel duration.** Timing a 50-microsecond kernel
  with coarse host clocks, or through launch overhead comparable to the kernel
  itself, measures the launcher. Use device-side timing or amortize over many
  launches and say which you did.
- **Shape cherry-picking.** Reporting the three shapes where your kernel wins,
  from a sweep of thirty, is a marketing document. Publish the full sweep;
  Chapter 18's capture-signature discipline exists partly so the served shapes
  are enumerable and therefore benchmarkable.
- **No error accounting.** A fused kernel that runs faster while changing
  accumulation order has changed the product. The layer's own table demands
  "latency *and numerical error*" — treat max-abs and distributional deltas as
  first-class outputs, not footnotes.

### Estimating capacity before measuring

Every serving team asks some version of the same question in a design review:
*roughly what will this configuration sustain?* Benchmarking answers it
eventually; arithmetic answers it now, well enough to know whether the plan
is plausible. Walk the whole estimate for one Atlas TP4 island using only
frozen and declared numbers.

**Decode throughput.** A model step reads the 35 GB weight shard plus each
active sequence's state. Against a declared memory path of ~3 TB/s, weights
alone imply `35 / 3 ≈ 12 ms` of streaming; adding 32 sequences at 2,000
context tokens (`32 × 2000 × 80 KiB ≈ 5 GiB`, about 1.7 ms more) sets a
roofline floor near 14 ms. Atlas's declared step is 45 ms — about a third of
the roofline — which is what real engines cost once launch overhead, attention
addressing, sampling, and collectives join the streaming. Take the 45 ms as
given: throughput is `32 tokens / 0.045 s ≈ 700` output tokens per second per
replica at batch 32.

**Concurrency.** KV capacity caps resident sequences before speed does:
`35 GiB / 0.61 GiB` per 8,000-token sequence ≈ 57, so batch 32 sits at 56
percent occupancy — headroom, not accident, given Chapter 6's preemption
arithmetic.

**Cross-check with utilization.** The same step in arithmetic terms is
`2 × 70 GFLOP × 32 ≈ 4.5 TFLOP` per 45 ms across four accelerators of ~1
PFLOPS-class peak: roughly two to three percent MFU, exactly the decode
ceiling Chapter 4 derived from intensity. And the frozen prefill cost
back-solves instructively: `prefill_ms(2000)` implies moving about 280 TFLOP
in 90 ms on the same island — near three-quarters of peak, an aggressive
large-batch figure that says Atlas's constants describe a *well-tuned*
system. When your measured prefill MFU lands far below that, the gap is a
diagnosis queue, not a mystery.

**Assemble.** At ~200 output tokens per response, decode alone sustains
about `700 / 200 ≈ 3.5` requests per second per replica; applying Chapter 2's
operating-utilization discipline and TTFT admission trims that to a few
qualifying requests per second — within a factor of two of Chapter 24's
declared operating point, which is precisely the accuracy band a
pre-benchmark estimate should claim. The estimate's real products are the
*constraints*: concurrency bounded by KV bytes, throughput by step time,
TTFT by prefill-plus-queue, and each bound naming the knob that would move
it. Benchmarks then refine numbers you can already defend; without the
estimate, they refine numbers you cannot.

## Reproduce the workload

A benchmark card should record arrival process, input and output length
distributions, prefix reuse, modality, priority, sampling parameters,
concurrency, and cache state. Preserve important correlations — real traffic
couples input length to output length (long documents get long answers) and
couples arrival bursts to working hours. A benchmark that samples each
distribution independently measures a workload that exists nowhere.

Use an open-loop generator for externally driven traffic and overload studies.
Use a closed-loop generator when modeling a fixed population of clients that
wait before issuing more work. Label the choice.

Warm and cold tests answer different questions. A cold test includes model
loading, compilation, graph capture, and empty caches. A steady-state test
should define its warm-up and confirm that compilation or allocation is no
longer changing the system — Chapter 18's capture-once discipline means a
well-behaved engine converges, but a benchmark that starts measuring before
the last graph is captured is measuring the compiler, not the service.

### Open loop, closed loop, and the knee they find

The two arrival processes find different knees, and the difference is
arithmetic, not taste. Little's law (`Q = λW`, Appendix A) says a closed-loop
population of `Q = 8` clients facing mean end-to-end latency `W = 1.5 s`
offers at most `λ = Q/W ≈ 5.3` requests per second. Now degrade the service:
latency doubles to 3 s. The closed loop's offered load *falls* to `8/3 ≈ 2.7`
requests per second — the clients politely back off exactly when you wanted to
see the failure, and measured throughput degrades gracefully all the way into
collapse. An open loop holding `λ` at 8 keeps the pressure on: queue age grows
without bound, TTFT breaches, and the admission controller's rejection path —
the one production will actually exercise — finally runs. Overload behavior is
a property of the *offered* load, so the experiment that studies it must pin
the offer, not the population. Closed loop remains the right model for fixed
client pools (an internal batch consumer with eight workers); the error is
using it to certify a service whose callers are the open internet.

### What a trace replay must preserve

Trace replay — SGLang's `use_trace_timestamps` mode, scaling recorded arrival
times by a `slowdown_factor` — is the most faithful generator and has its own
validity conditions. The replayer must actually *keep up*: if issuing a request
takes longer than the next trace timestamp allows, arrivals silently compress
and the offered load drifts above the value on the card. Assert on issued-versus-
scheduled timestamps rather than trusting the sleep loop. Prefix-cache state
carries across requests within a run, so replay order *is* part of the
workload — shuffling a trace changes reuse even though its length distribution
is untouched. Multi-turn conversations are another preserved correlation: a
follow-up turn's prompt includes the prior answer, so breaking conversations
into independent requests changes both prefix structure and input-length
distribution at once. vLLM's benchmark package keeps a dedicated
[`multi_turn`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/benchmarks/multi_turn)
directory for exactly this reason. Finally, record whether the harness pinned
sampling (`temperature: 0.0` in vLLM's default payload) or replayed production
sampling parameters — the first isolates system performance, the second is
more faithful to output-length variance, and the card should say which.

## Measure latency-bounded throughput

Increase offered load until one or more SLOs fail. Plot goodput rather than
publishing one throughput point. The curve reveals saturation and collapse.

Learn to read the curve's three regions. At low load, goodput tracks offered
load one-to-one and latencies sit near their floor — the system is
transporting, not queueing. Near the knee, goodput growth flattens as queue
delay consumes SLO budgets; this is the operating region, and production
should sit below it with headroom sized for arrival bursts. Past the knee,
goodput *falls* as offered load rises — queues lengthen, TTFT breaches spread,
and every additional request makes the others worse. The single most useful
number from a sweep is not peak goodput but the distance between the chosen
operating point and the knee, because that distance is the service's tolerance
to a traffic surprise.

Report latency distributions, not only averages. Keep errors, cancellations,
and timeouts in the accounting. A request that disappears from the sample when
it times out makes the service look better as it fails — the timeout filter is
a machine for converting overload into flattering numbers.

Benchmark standards such as
[MLPerf Inference](https://docs.mlcommons.org/inference/) demonstrate the value
of defined scenarios, quality targets, and run rules for comparable results.
Your production benchmark will use different models and traffic, but it should
be equally explicit about the contract.

### Inside an open-loop harness

The pinned serving benchmarks show how much contract hides inside a
load-generator script. vLLM's
[`vllm/benchmarks/serve.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/benchmarks/serve.py)
at the pinned SHA generates arrivals in `get_request`: with `burstiness = 1.0`
intervals are exponential — "it follows exponential distribution" — and the
general case samples from a gamma distribution whose shape *is* the burstiness
parameter, so one dial moves arrivals between bursty and uniform. Two details
reward attention. First, the generator *precomputes* the whole arrival
schedule, then rescales it: the comment notes that summed gamma draws "would
have 1-2% gap from target_total_delay_s," and normalization "close[s] the gap
for stabilizing the throughput data from different random seeds" — even the
arrival process is calibrated, because an unnormalized generator would report
throughput variance that belongs to the harness, not the system. Second,
`self_timed` mode abandons synthetic arrivals entirely and replays recorded
trace timestamps, scaled by a slowdown factor — the workload *is* the arrival
process, so the most faithful generator is the one that stops generating.

SGLang's
[`sglang/benchmark/serving.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/benchmark/serving.py)
adds a measurement-validity correction this book's Chapter 11 makes
predictable: with speculative decoding, one streamed chunk can carry several
accepted tokens, so raw per-chunk inter-token latencies overstate per-token
latency. Its `use_retokenized_itl` path divides each chunk's ITL by the
retokenized token count of that chunk's text (`adjusted_itl = itl /
num_tokens`) and expands the series accordingly. A harness that ignored the
bundling would "measure" speculative decoding as an ITL regression.

Both harnesses agree on what counts as a request. In vLLM's
[`backend_request_func.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/benchmarks/backend_request_func.py),
a stream that returns HTTP 200 but never delivers a token-bearing chunk is
recorded as *failed* — "Never received a valid chunk to calculate TTFT. This
response will be marked as failed!" — and the payload pins
`temperature: 0.0`, because sampling variance is not what is under test.
Goodput is a per-request conjunction: `is_good_req = all(...)` over every
configured SLO, so a request that meets TTFT but misses TPOT counts as
failed for goodput purposes. Chapter 21's rule — "both returned HTTP 200 is
not conformance" — is the same discipline pointed at correctness; here it is
pointed at speed.

## Protect semantic equivalence

Two systems are comparable only if they perform equivalent work. Check model
weights, precision, tokenizer, template, context limit, sampling, stop rules,
structured output, and output quality.

Quantization and speculative decoding require quality or distribution checks.
Prefix caching requires output equivalence. Different truncation policies can
make one engine appear faster by processing less input. Record accepted and
rejected speculative tokens separately from user-visible output — the
retokenized-ITL correction above is exactly why: the acceptance statistics are
the mechanism's evidence, while the retokenized series is the user's
experience, and collapsing one into the other loses both.

When strict equality is not expected, define the evaluation and acceptable
change before seeing the result. "Within 1 point on the quality task" chosen
after the run is not a gate; it is a rationalization with a number attached.

Design the gate to match the mechanism's risk. Greedy decoding under an
algebraically identical kernel should match near-exactly, with divergence only
from floating-point non-associativity — so the gate is a tolerance on token
divergence position, not a task score. Quantization changes the product by
design, so the gate is a task-metric regression bound agreed before the run,
plus a distributional check on the outputs Chapter 10 recommends. Speculative
decoding should be *output-preserving* under its verification contract — any
accepted-token distribution shift is a bug in the drafter or verifier, not a
quality trade-off to be weighed. Structured-output changes gate on Chapter
21's conformance suite rather than on task metrics, because the property under
test is syntactic and behavioral, not statistical. One gate per risk; a
generic "quality looks fine" gate catches none of them.

## Control the environment

Record engine and model commits, container digest, compiler, drivers, firmware,
device model, power settings, CPU and NUMA placement, interconnect topology,
and relevant environment variables. The test of adequacy is rebuildability: a
reader with the card should be able to construct a byte-identical system. A
version string does not satisfy this — the same release with different launch
flags is a different system, and flag drift between "identical" runs is a
classic source of irreproducible comparisons. Note other workloads on shared
hardware — a neighbor's training job can move your p99 by more than the
optimization under test.

Run enough repetitions to characterize variance. Randomize experiment order
when temperature or shared infrastructure can drift: thermal state is a
slow variable, and a fixed A/B-A/B order lets it masquerade as a treatment
effect. Report confidence intervals or the raw distribution instead of excessive
decimal precision.

Do not tune one system extensively while leaving another at defaults. Either
compare documented defaults for a stated purpose or give each system a fair
tuning budget and publish the configurations.

### How many repetitions, and of what

Repetition budget follows from the statistic being claimed, and the demands are
not symmetric. A mean stabilizes quickly; a tail does not. Estimating p99
latency within a useful tolerance requires observing many samples beyond it —
with 100 requests, the p99 estimate is effectively "the worst request," which
is one sample wearing a percentile costume. As a declared working heuristic:
tail percentiles need thousands of requests per cell to move less than the
effect you are testing for, so either collect that volume or claim a lower
percentile honestly. This is also why per-request raw events (the card's
outputs line) matter more than the summary: pooled across runs they support
bootstrap intervals, while pre-aggregated summaries cannot be re-analyzed.

Structure the repetitions as blocks. Run baseline-and-candidate back-to-back
within each block, then repeat blocks in randomized order — blocking absorbs
the slow drift (thermals, background load) into comparisons *within* a block,
where both systems see the same environment. Report the spread across blocks;
if blocks disagree beyond their internal noise, the card's second-day
procedure has already told you what to inspect next.

## Profile after locating the regime

A profiler explains a result; it does not define the workload. First identify
the batch sizes and load range where behavior changes. Then capture CPU and GPU
timelines, kernel counters, memory activity, collectives, and network traffic in
that regime.

Look for idle gaps, synchronization, unexpected copies, graph fallbacks,
padding, imbalance, and queue transitions. Connect every low-level observation
back to a service metric. "GPU utilization increased" matters only if useful
output or cost improved — utilization is a *diagnostic*, and optimizing it
directly produces busy systems that serve no one.

Capture has a cost, and the cost perturbs the thing being measured: timeline
tracing lengthens steps, and a capture run is therefore a *different*
benchmark from the un-instrumented run that produced the headline number.
Treat profiling runs as explanatory evidence attached to a regime, never as
the source of the performance claim — Chapter 5's step anatomy was built from
exactly this kind of capture, and it explains the 45 ms step without being the
place the 45 ms was certified.

## Publish negative and conditional results

An optimization that loses under high concurrency or low prefix reuse is
valuable information. It helps readers learn the boundary of the mechanism.

Use conditional language:

> On this model and device, with the measured shape distribution, configuration
> A increased TTFT-qualified goodput by 18 percent. It regressed the low-load
> median because graph padding dominated below eight active sequences.

This result is more durable than a framework ranking. The 18 percent will not
transfer to another reader's traffic, but the *shape* of the boundary — where
padding stops being amortized — will, and the next reader can test their own
position relative to it.

Negative results also compound only if they survive publication: attach the
card, raw events, and analysis revision to the report, because a negative
result without its evidence cannot be re-tested when a reader's conditions
differ. An optimization retired on an unarchived benchmark will be re-proposed,
re-benchmarked, and re-rejected within the year at full cost.

## Worked example: make the claim falsifiable

"Configuration B is 18 percent faster" is not a benchmark claim. A useful claim
states that B improves TTFT-qualified goodput for the Atlas document trace,
under an open-loop arrival process, while passing the same quality gate. It
pins the model, engine commit, container, hardware topology, workload hash,
warm-up, cache state, error accounting, and analysis revision.

Walk the claim against the card in Appendix G. The SLO line — success, TTFT
at most 600 ms, every ITL at most 150 ms, valid output — makes goodput
computable: a request is good only if all four hold, so at an offered 8
requests per second, "goodput of 6.2" means 6.2 requests per second passed
*every* gate. Attribution then comes from the raw events, not intuition:
suppose (declared example) 1.1 requests per second missed TTFT, 0.4 missed an
ITL sample, and 0.3 errored — the candidate's TTFT mechanism is worth
pursuing, the error path is a bug regardless of speed, and the ITL tail needs
one more sweep near the knee before any conclusion. The workload line's prefix
distribution matters because Atlas's routing hypothesis lives or dies on reuse:
run the sweep at low and high reuse, not one blend, or the result cannot say
whether routing helped the cached or the uncached population.

Run baseline and candidate in randomized order at several offered loads. Keep
timeouts and errors in the denominator. If a second-day result moves, inspect
temperature, clock policy, cache warmth, artifact hashes, and background
traffic rather than averaging two regimes into one misleading number — the
card's method section exists precisely so the second-day run differs from the
first in at most the ways it lists. Every field on the card earns its place by
naming a way the comparison could silently break; a field you cannot connect
to a failure mode is decoration, and a failure mode with no field is an
unprotected flank.

## Practice: complete a benchmark card

Write the full card for a candidate scheduler that claims higher goodput on the
traces from Chapter 2. Include commands, configuration, system identity,
quality checks, repetitions, raw event schema, analysis revision, and second-
day reproduction procedure.

State the exact claim the evidence could falsify. A complete example appears in
[Appendix G](../appendices/g-worked-solutions.md#22-benchmark-card).

Once a service is deployed, the experiment continues under real traffic.
Chapter 23 covers the signals and operating practices that keep it interpretable.
