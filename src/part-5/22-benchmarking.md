# 22. Benchmarking and Performance Science

A benchmark is an experiment designed to answer a decision. Without the
decision, it becomes a number generator.

“Which engine is fastest?” is too broad. A useful question is narrower:

> For our model, hardware, request distribution, and TTFT/ITL targets, does
> prefix-aware routing increase goodput enough to justify its control-plane
> complexity?

That question tells you what to hold constant, what to vary, and which result
matters.

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
per second, you cannot test it.

Also write a falsification condition. If smaller chunks lower ITL but reduce
goodput after TTFT constraints are included, the proposed configuration did not
achieve its goal.

## Use the right benchmark level

A kernel microbenchmark isolates one operation and is ideal for comparing
implementations across shapes. An engine-step benchmark includes input
preparation, kernels, graphs, and collectives. An end-to-end load test includes
queues, routing, preprocessing, streaming, and clients.

Each level removes noise and context. Start at the lowest level that can answer
the question, then verify at the service boundary. A kernel speedup that
vanishes at the engine step is still useful diagnostic evidence; it is not an
end-to-end performance result.

## Reproduce the workload

A benchmark card should record arrival process, input and output length
distributions, prefix reuse, modality, priority, sampling parameters,
concurrency, and cache state. Preserve important correlations.

Use an open-loop generator for externally driven traffic and overload studies.
Use a closed-loop generator when modeling a fixed population of clients that
wait before issuing more work. Label the choice.

Warm and cold tests answer different questions. A cold test includes model
loading, compilation, graph capture, and empty caches. A steady-state test
should define its warm-up and confirm that compilation or allocation is no
longer changing the system.

## Measure latency-bounded throughput

Increase offered load until one or more SLOs fail. Plot goodput rather than
publishing one throughput point. The curve reveals saturation and collapse.

Report latency distributions, not only averages. Keep errors, cancellations,
and timeouts in the accounting. A request that disappears from the sample when
it times out makes the service look better as it fails.

Benchmark standards such as
[MLPerf Inference](https://docs.mlcommons.org/inference/) demonstrate the value
of defined scenarios, quality targets, and run rules for comparable results.
Your production benchmark will use different models and traffic, but it should
be equally explicit about the contract.

## Protect semantic equivalence

Two systems are comparable only if they perform equivalent work. Check model
weights, precision, tokenizer, template, context limit, sampling, stop rules,
structured output, and output quality.

Quantization and speculative decoding require quality or distribution checks.
Prefix caching requires output equivalence. Different truncation policies can
make one engine appear faster by processing less input. Record accepted and
rejected speculative tokens separately from user-visible output.

When strict equality is not expected, define the evaluation and acceptable
change before seeing the result.

## Control the environment

Record engine and model commits, container digest, compiler, drivers, firmware,
device model, power settings, CPU and NUMA placement, interconnect topology,
and relevant environment variables. Note other workloads on shared hardware.

Run enough repetitions to characterize variance. Randomize experiment order
when temperature or shared infrastructure can drift. Report confidence
intervals or the raw distribution instead of excessive decimal precision.

Do not tune one system extensively while leaving another at defaults. Either
compare documented defaults for a stated purpose or give each system a fair
tuning budget and publish the configurations.

## Profile after locating the regime

A profiler explains a result; it does not define the workload. First identify
the batch sizes and load range where behavior changes. Then capture CPU and GPU
timelines, kernel counters, memory activity, collectives, and network traffic in
that regime.

Look for idle gaps, synchronization, unexpected copies, graph fallbacks,
padding, imbalance, and queue transitions. Connect every low-level observation
back to a service metric. “GPU utilization increased” matters only if useful
output or cost improved.

## Publish negative and conditional results

An optimization that loses under high concurrency or low prefix reuse is
valuable information. It helps readers learn the boundary of the mechanism.

Use conditional language:

> On this model and device, with the measured shape distribution, configuration
> A increased TTFT-qualified goodput by 18 percent. It regressed the low-load
> median because graph padding dominated below eight active sequences.

This result is more durable than a framework ranking.

## Worked example: make the claim falsifiable

“Configuration B is 18 percent faster” is not a benchmark claim. A useful claim
states that B improves TTFT-qualified goodput for the Atlas document trace,
under an open-loop arrival process, while passing the same quality gate. It
pins the model, engine commit, container, hardware topology, workload hash,
warm-up, cache state, error accounting, and analysis.

Run baseline and candidate in randomized order at several offered loads. Keep
timeouts and errors in the denominator. If a second-day result moves, inspect
temperature, clock policy, cache warmth, artifact hashes, and background
traffic rather than averaging two regimes into one misleading number.

## Practice: complete a benchmark card

Write the full card for a candidate scheduler that claims higher goodput on the
traces from Chapter 2. Include commands, configuration, system identity,
quality checks, repetitions, raw event schema, analysis revision, and second-
day reproduction procedure.

State the exact claim the evidence could falsify. A complete example appears in
[Appendix G](../appendices/g-worked-solutions.md#22-benchmark-card).

Once a service is deployed, the experiment continues under real traffic.
Chapter 23 covers the signals and operating practices that keep it interpretable.
