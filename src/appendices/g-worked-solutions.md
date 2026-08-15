# Appendix G. Worked Solutions

The exercises use a continuing fictional service called **Atlas**, an
enterprise research assistant. Atlas serves a dense decoder with approximately
70 billion parameters. The baseline uses BF16 weights, 80 transformer layers,
8 KV heads, and head dimension 128. Traffic is 70 percent short interactive
questions, 20 percent document questions, and 10 percent long research jobs.
Many requests share tenant instructions or uploaded documents.

These are worked engineering answers, not universal configurations. Where an
exercise would normally require measurements, the solution states assumptions
and shows the decision process instead of inventing results.

## 1. Request trace

Assume a user sends a 6,000-token document question and asks for at most 300
output tokens. A defensible trace is:

```text
client
  -> edge queue
  -> API validation and tenant lookup
  -> tokenizer queue
  -> cache-aware router
  -> engine admission queue
  -> prefix lookup and KV allocation
  -> prefill steps
  -> decode steps
  -> detokenizer and stream buffer
  -> client
```

The request record is owned by the API until accepted by the engine, then by
the engine until final cleanup. The block manager owns KV allocation. The
router owns only a possibly stale location hint; it does not own the cached
blocks. Cancellation can arrive while queued, during prefill, during decode, or
after a token has entered the output buffer. In every case the engine stops
future scheduling, marks in-flight work discardable, and releases blocks only
after the last GPU user finishes.

The least visible delay in this example is the cache-aware routing decision.
The chosen replica saves 4,000 prompt tokens but has 450 ms of queued prefill;
an idle replica could recompute them in 280 ms. Locality is therefore a loss.
The trace needs both `queue_age_at_assignment` and `matched_prefix_tokens` to
make that diagnosis possible.

A complete answer names the owner and queue at every boundary. A box-and-arrow
diagram without those labels does not reveal cancellation safety or latency.

## 2. Workload traces and goodput

Construct 100 requests with 100,000 total input tokens and 20,000 total output
tokens in each trace:

| Trace | Construction | Expected pressure |
| --- | --- | --- |
| Even | 100 × 1,000 input, 200 output; fixed spacing | stable batching |
| Bursty | 80 short and 20 long; five arrival bursts | queue and prefill interference |
| Conversational | ten shared 8,000-token prefixes plus short turns | cache locality and pauses |

Use an open-loop arrival rate of 8 requests/s. Define a qualifying interactive
request as successful, TTFT at most 600 ms, no ITL above 150 ms, and valid
output. If 100 requests arrive over 12.5 seconds, 96 finish, and only 81 meet all
conditions, throughput is `96 / 12.5 = 7.68 requests/s` while goodput is
`81 / 12.5 = 6.48 requests/s`.

The even trace should have the narrowest latency distribution. The bursty
trace can have identical token throughput yet lower goodput because long
prefills delay active decoders. The conversational trace should benefit from
prefix reuse, but only if routing does not create a hot replica. A closed-loop
rerun will probably look healthier because slow responses reduce the offered
load; that is a property of the generator, not an engine improvement.

The correct report preserves each trace's length and arrival correlations. It
does not shuffle prompt lengths independently or average all traffic classes
into one latency number.

## 3. Model topology inventory

For the Atlas dense decoder, BF16 parameter storage is approximately
`70 billion × 2 bytes = 140 GB`, before allocator and runtime overhead. Its KV
state per token is:

```text
2 × 80 layers × 8 KV heads × 128 dimensions × 2 bytes = 327,680 bytes
```

That is 320 KiB per token across the model, or about 2.44 GiB for an 8,000-token
sequence. Tensor parallelism shards this state, but does not make the aggregate
bytes disappear.

For a hypothetical 8 × 7B expert model with two experts active per token, total
BF16 expert weights are roughly 112 GB plus shared attention and router
weights. Per-token arithmetic uses two experts, while expert parallel serving
adds dispatch and combine traffic. The topology inventory therefore includes
both total resident bytes and active bytes per token.

For a hybrid vision-language model, list the vision encoder, projection, dense
decoder, encoder outputs, and decoder KV state separately. A 2,048-token image
feature sequence reused across five questions is a candidate for encoder-output
caching or encoder disaggregation. Its transfer size, not the original JPEG
size, governs that boundary.

The serving conclusion is not “MoE is cheaper” or “hybrid is slower.” The dense
model is dominated by replicated or sharded weights and growing KV state. MoE
adds conditional placement. The hybrid adds a separately batchable encoder and
reusable feature state. Those facts determine which serving plans are legal.

## 4. Topology prediction

Assume two eight-GPU nodes. Each GPU has 80 GiB of device memory; links inside a
node are fast, and the inter-node network is slower. Place one four-way
tensor-parallel Atlas replica entirely inside each half-node. At 140 GB of BF16
weights, each rank holds roughly 35 GB before non-parameter overhead. The KV
state is also sharded: an 8,000-token request holds about 625 MiB per rank.

The initial prediction is:

| Work | Likely limit | Reason |
| --- | --- | --- |
| 8,000-token prefill | compute or attention memory traffic | large matrix work and quadratic attention traffic |
| batch-1 decode | device memory bandwidth plus TP latency | weights are read for little new-token work |
| 2.44-GiB KV move | slowest transfer edge | bulk state crosses device, host, or network boundaries |

Do not stripe one TP group across both nodes unless measurement shows the
inter-node collectives are acceptable. The arithmetic is unchanged, but each
layer now depends on the slower fabric.

Profile counters should be chosen to falsify the prediction: achieved compute
and HBM bandwidth for prefill; memory bandwidth, GPU gaps, and collective time
for decode; payload bytes, staging copies, and concurrent-link throughput for
the KV move. If decode shows low HBM traffic and large CPU gaps, the original
prediction was incomplete—the host or launch path is the actual limit.

## 5. Control and data paths

For one Atlas request, the control path is:

```text
submit -> validate -> admit -> schedule -> allocate -> execute -> finish
```

The data path is:

```text
text -> token IDs -> model tensors -> KV blocks -> logits -> token IDs -> text
```

They meet at several synchronization points. Admission waits for an ownership
decision so rejected work cannot allocate state. The model runner waits for the
scheduler's block table because attention must address the right physical
pages. The sampler waits for logits because the next token is a true dependency.
Cleanup waits for the completion event of in-flight GPU work so memory is not
reused early.

Other waits are candidates for overlap. Tokenizing request B can overlap model
execution for request A. Output processing for step `n` can overlap GPU work for
step `n + 1` if request-state updates are versioned. A cache write-back can be
asynchronous if eviction and failure do not make the only valid copy disappear.

The answer should attach an invariant to every wait. “The CPU waits for the
GPU” is description; “cleanup waits so no block can be reallocated while the
GPU still holds its address” is an engineering explanation.

## 6. Scheduler simulation

Use a token budget of 16 per step and enough memory for 40 live token-block
units. Four requests arrive:

| Request | Arrival | Prefill | Output | Priority |
| --- | ---: | ---: | ---: | ---: |
| A | 0 | 24 | 4 | normal |
| B | 0 | 4 | 8 | normal |
| C | 1 | 8 | 4 | high |
| D | 2 | 20 | 2 | normal |

With first-come-first-served and no chunking, A consumes the first two steps
and B's first token waits. With an 8-token prefill chunk, step 0 can schedule
eight tokens of A and four of B. B enters decode earlier; later steps mix one
decode token for B and C with bounded chunks of A or D.

A reasonable priority-with-aging score is:

```text
score = base_priority + 0.05 × waiting_steps
```

Always reserve enough of the step budget for active decoders before scheduling
prefill. Reject D if its predicted completion is already beyond its deadline or
if admitting its state would force higher-priority C to preempt.

The simulator should use a measured step-time table such as `(decode batch,
prefill tokens) -> milliseconds`. Counting every simulated step as equal would
hide the interference being studied. Report TTFT and deadline-qualified
goodput per class, not only the number of scheduled tokens.

## 7. KV-cache correctness matrix

Begin with a known 512-token prefix and change one identity dimension at a
time:

| Change | Reuse? | Reason |
| --- | --- | --- |
| token 511 differs | first 511 tokens only | content identity ends before mismatch |
| adapter differs | no, unless compatibility is proven | activations depend on adapter weights |
| model version differs | no | keys and values were produced by different weights |
| tenant differs | policy-dependent, default no | isolation can be stricter than numerical identity |
| image feature differs | no beyond insertion point | multimodal positions depend on the feature |
| block layout differs | possible after validated remap | physical address is not semantic identity |

Now cancel a writer after the GPU has produced a partial final block but before
publication. The block remains private and pinned until the completion event;
it is then discarded or published atomically according to the cancellation
policy. A branch shares only sealed full blocks. Its partial tail is copied on
write. Eviction removes the reusable index entry first, then frees storage only
when the reference count reaches zero.

Three invariants make the test decisive: no lookup returns an unpublished
block; no physical block is writable by two logical branches; and every
terminal request eventually releases its references. Output equivalence should
be checked against a cache-disabled run, not merely against another cached run.

## 8. Kernel evaluation

Suppose a new paged-attention kernel is 22 percent faster than the baseline for
batch 32, context 4,096, head dimension 128. The isolated result is only level
one.

At engine-step level, add metadata construction, block-table transfer, layout
conversion, sampling, and synchronization. If the old step takes 5.0 ms, of
which attention is 2.0 ms, a 22 percent attention improvement saves at most
0.44 ms before new overhead. A 0.3 ms conversion leaves the step only 2.8
percent faster: `(5.0 - 0.14) / 5.0`.

At workload level, assume the kernel requires 64-token pages instead of
16-token pages. Short prefixes now waste more cache tail space and exact reuse
ends at coarser boundaries. Lower cache capacity can create preemption or
recomputation that erases the 0.14 ms step gain.

The decision is therefore conditional: enable the kernel only for supported
shapes or page layouts until end-to-end goodput improves. Correctness tests
cover single-token decode, non-multiple context lengths, partially filled
pages, empty sequences, mask boundaries, and tolerances against a trusted
implementation. A fast result for one rectangular shape does not justify a
global backend switch.

## 9. Compilation and graph buckets

Assume one minute of Atlas traffic produces decode batch sizes:

```text
1: 8%, 2-4: 17%, 5-8: 31%, 9-16: 29%, 17-32: 15%
```

Candidate graph buckets are 1, 4, 8, 16, and 32. Dispatch rounds a batch up to
the smallest compatible bucket. Batch 9 therefore uses bucket 16 and executes
seven padded slots. The trace should record both the requested and replayed
shape so padding is visible.

Compare three regimes after a separate cold-start measurement. If eager uses
1.1 ms CPU launch time and 4.0 ms GPU time at batch 8, graph replay that reduces
CPU time to 0.2 ms but adds 0.15 ms padding/dispatch work improves the step from
5.1 to 4.35 ms. At batch 9 replaying bucket 16 might add enough padded GPU work
to lose.

The response is not to disable graphs. Add a bucket near a frequent costly gap,
allow eager fallback for rare shapes, and set a memory budget for captured
artifacts. Report compilation time separately, plus artifact count, bucket hit
rate, padding ratio, fallback rate, graph memory, and SLO-qualified goodput.

## 10. Quantization decision

Compare BF16, weight-only INT4, and FP8 weights plus FP8 KV state. Use the same
Atlas request trace and product evaluation set.

| Axis | BF16 | INT4 weight-only | FP8 weight + KV |
| --- | --- | --- | --- |
| weight bytes | baseline | about one quarter plus scales | about one half plus scales |
| KV capacity | baseline | unchanged | roughly doubled before overhead |
| likely benefit | reference quality | model fit and decode traffic | model and long-context capacity |
| principal risk | cost | dequantization/kernel shape | calibration and attention stability |

Do not infer speed from the table. Measure representative prefill and decode
shapes. If INT4 saves weight bandwidth but its kernel performs poorly at batch
1, it may increase interactive ITL. If FP8 KV doubles capacity, it may improve
goodput by avoiding preemption even when one isolated attention call is
unchanged.

Gate quality using product tasks, tool-call validity, long-context retrieval,
rare languages, and log-probability drift. A sensible decision record might
choose FP8 for the long-document tier, retain BF16 for sensitive evaluation,
and reject INT4 until the target small-batch kernel improves. The answer names
the binding service constraint; “FP8 won” is incomplete.

## 11. Speculation break-even

Let ordinary target decode cost 8 ms per token. A speculative step proposes
four tokens, costs 3 ms to draft, and 9 ms to verify. If it accepts an average
of 3.2 tokens, cost per accepted token is `(3 + 9) / 3.2 = 3.75 ms`, a strong
win before system overhead.

On an unpredictable prompt, suppose only 1.3 tokens are accepted. Cost becomes
`12 / 1.3 = 9.23 ms`, slower than ordinary decode. At high concurrency, the
draft model's memory may also reduce target KV capacity, while verification
uses larger shapes that contend with other requests.

The serving policy should estimate expected accepted tokens and compare:

```text
draft cost + verify cost + capacity cost
        versus
ordinary cost × expected accepted tokens
```

Use a conservative threshold and turn speculation off for short remaining
outputs, low historical acceptance, memory pressure, or graph-incompatible
shapes. Verify output distribution equivalence for the algorithm in use; equal
tokens under one random seed are useful debugging evidence but not the whole
distributional contract.

## 12. Two parallel plans

Use eight GPUs in one fast-link island for one Atlas replica.

**Plan A:** tensor parallel size 8. Every layer shards large matrix operations
and performs layer-frequency collectives across all eight ranks. Weight and KV
memory per rank are smallest, and there is no pipeline bubble, but decode pays
the widest collective latency.

**Plan B:** pipeline size 2 and tensor parallel size 4. Two groups of four own
half the layers. TP collectives stay within smaller groups; one activation
tensor crosses the stage boundary. A single interactive request creates a
pipeline bubble, while enough concurrent microbatches can fill both stages.

For a hidden width of 8,192 and BF16 activation, one token's stage-boundary
payload is about 16 KiB per sequence before batching. TP collective volume is
implementation-dependent, so the answer must derive it from the chosen sharding
algorithm rather than assert one universal formula.

Predict Plan A for low-concurrency prefill if its all-rank fabric is excellent
and minimizing bubbles matters. Plan B may win for sustained throughput when
four-rank collectives are materially cheaper and concurrency fills the
pipeline. Measure prefill and decode separately; one winner is not required.

## 13. Expert trace and placement

Assume one MoE layer has eight experts on four ranks, two experts per rank. A
decode step routes 64 tokens with counts:

```text
E0 22, E1 14, E2 7, E3 6, E4 5, E5 4, E6 3, E7 3
```

With contiguous placement `(E0,E1)` on rank 0, that rank receives 36 expert
assignments while rank 3 receives 6. The layer finishes at the hot rank.
Moving E1 to rank 3 produces loads 25, 13, 9, and 17 if paired carefully. The
maximum falls, although remote dispatch paths change.

If the activation width is 8,192 in BF16, one routed copy is 16 KiB. For top-2
routing, 64 tokens create 128 assignments, or roughly 2 MiB of activation
payload before protocol overhead for dispatch and another combine transfer.
The trace should count assignments, not unique input tokens.

An EPLB update is worthwhile only if reduced straggler time exceeds weight-copy
and cache disturbance. Apply a generation number: new batches use the new
placement only after every rank acknowledges it; old batches finish under the
old map. Keep a fallback copy during the transition if memory allows.

## 14. Prefill/decode split

Assume measured service models:

```text
prefill_ms(tokens) = 20 + 0.035 × tokens
decode step at target batch = 45 ms
KV transfer = 12 ms setup + bytes / 22 GiB/s
```

Atlas KV state is 320 KiB per token. A 6,000-token prompt creates about 1.83
GiB, so its idealized transfer takes roughly 85 ms plus 12 ms setup. The
prefill itself is approximately 230 ms. Transfer is therefore a material stage,
not a rounding error.

Disaggregation can still win if decode no longer waits behind long prefills and
the two pools are independently saturated. Model three queues: prefill,
transfer, and decode. Admission requires a predicted decode slot, not only an
idle prefill worker.

Use conditional placement: keep short uncached prompts colocated; disaggregate
long prompts when the expected reduction in decode interference exceeds
transfer and extra queueing. On transfer failure, discard the unpublished
destination state and either retry within the deadline or recompute. Report
stage utilization because low end-to-end latency obtained with a mostly idle
pool may be economically unacceptable.

## 15. Distributed prefix lifecycle

Give the prefix an identity derived from model version, tokenizer, adapter,
tenant namespace, token sequence, position scheme, and state format. GPU A
creates blocks privately, seals them after execution, and publishes metadata
only after the data is readable.

Host backup takes a read reference and copies the sealed blocks. A successful
checksum and generation match publish the host location. Remote metadata may
advertise that location, but it is a hint: GPU B obtains a lease, revalidates
identity, reserves destination blocks, transfers, verifies, and only then
inserts them into its local index.

If B cancels mid-transfer, the destination blocks stay unpublished and are
released after the copy completion or abort event. A metadata timeout causes a
miss and recomputation, not indefinite request blocking. Invalidation removes
new lookup visibility first; existing readers finish through references or
leases; physical deletion follows when references reach zero.

For a 1-GiB prefix that saves 180 ms of prefill but needs 70 ms to load, the
gross request saving is 110 ms. Include queueing and the opportunity cost of 1
GiB on the destination before deciding to promote it. A hit counter alone
cannot express that value.

## 16. Cache-aware routing

Assume three replicas. R0 has a 4,000-token prefix but 300 ms of queued work;
R1 is idle without the prefix; R2 has half the prefix and 100 ms queued.
Recomputation costs 0.06 ms per missing prompt token.

Estimate completion contributions:

| Replica | Queue | Missing-prefill work | Combined estimate |
| --- | ---: | ---: | ---: |
| R0 | 300 ms | 0 ms | 300 ms |
| R1 | 0 ms | 240 ms | 240 ms |
| R2 | 100 ms | 120 ms | 220 ms |

R2 wins this simplified decision. Cache-only routing would choose R0 and least-
queue routing would choose R1; both ignore useful information. Add transfer
cost, uncertainty, adapter availability, and deadline risk in a real score.

When one prefix becomes very hot, one cached replica becomes a queue hotspot.
Replicate the prefix only when predicted avoided recomputation and queue relief
justify its memory. Use hysteresis so placement does not oscillate with every
small popularity change. The simulator should delay telemetry deliberately;
perfect instantaneous queue knowledge would make the router unrealistically
powerful.

## 17. Multimodal first-output path

For a repeated image question, suppose the first request has this trace:

| Stage | Duration |
| --- | ---: |
| receive and fetch | 35 ms |
| decode and preprocess | 28 ms |
| encoder queue | 40 ms |
| vision encoder | 115 ms |
| feature transfer | 12 ms |
| language queue and prefill | 190 ms |
| first decode token | 45 ms |

TTFT is 465 ms. A processed-image cache saves only the 28 ms preprocessing
stage. An encoder-output cache saves preprocessing, encoder queue, encoder, and
possibly feature creation—183 ms here—at the cost of retaining a larger,
model-version-specific tensor. Full language-prefix reuse may save still more
but is invalid if the question tokens occur inside the reusable prefix boundary.

The second question must retain the same media identity, preprocessing
configuration, model version, and feature layout. Compare output with a cache-
disabled request. If disaggregating the encoder adds a 35 ms transfer instead
of 12 ms but removes an 80 ms queue at the language worker, it improves TTFT;
for uncached tiny images, the extra boundary may lose.

## 18. Diffusion timeline

Assume an image pipeline spends 18 ms in text encoding, 30 denoising steps at
24 ms each, 55 ms in latent decoding, and 22 ms postprocessing. Total service
time is `18 + 720 + 55 + 22 = 815 ms`; denoising is the clear target.

A cache that skips equivalent work in 10 steps saves at most 240 ms before
lookup and correction cost. It must be evaluated with prompt classes and image
quality, not only cache hit rate. At double spatial resolution, latent work can
grow far faster than the text stage, so the bottleneck conclusion should be
retested.

For disaggregation, text embeddings are small and easy to transfer, while the
latent entering the decoder can be large. Separate a stage only if independent
scaling, reuse, or batch compatibility offsets the transfer and queue. A good
timeline records every denoising step, graph bucket, cache action, synchronization,
and stage-boundary byte count.

Report paired blinded samples and a declared quality metric. A latency win that
changes composition or temporal consistency is a different product setting,
not a free acceleration.

## 19. Policy update transaction

Every rollout carries `policy_version = 41`. The trainer produces version 42 in
staging storage with a manifest of tensors, shapes, dtypes, and checksums.
Inference ranks stop admitting version-41 groups, finish or abandon them by
policy, enter sleep or update state, and copy version 42 into inactive buffers.

Each rank validates the manifest and reports `prepared(42)`. Only after all
ranks prepare does the coordinator publish commit generation 42. Ranks swap
buffers, invalidate version-dependent KV and graph artifacts, run a health
forward pass, and report ready. New rollout admission then resumes with version
42.

If rank 3 fails halfway through copy, no commit is published. Prepared ranks
retain version 41 as active and discard or retry their inactive buffers. The
service never constructs one tensor-parallel group from mixed versions.

When the trainer is delayed, bound rollout work by both queue bytes and maximum
policy lag. Stop admission before trajectories grow without bound. The trainer
may accept completed version-41 groups if the algorithm allows that lag; the
serving layer must not invent the rule. Log token IDs, masks, sampling state,
and log-probability semantics so training can reproduce or deliberately trust
the rollout calculation.

## 20. Ten-second conversation

One coherent timeline is:

```text
0.0-2.4  user speech; ASR partials at 0.8, 1.5, 2.2
2.4-2.6  endpoint stabilization and final transcript
2.6-2.8  LLM routing and prefill
2.8-3.0  first text; tool call emitted
3.0-3.6  tool runs; assistant sends a brief holding phrase
3.6-3.8  final response begins; TTS buffers first audio
3.8-5.1  assistant audio plays
5.1      user interrupts; turn generation changes from 7 to 8
5.18     playback is silent
5.4      stale TTS chunk for generation 7 arrives and is discarded
```

The end-of-turn-to-first-audio budget is 1.4 seconds here: 200 ms endpointing,
200 ms language startup, 600 ms tool time overlapped with a holding phrase, and
200 ms TTS plus buffer, with remaining transport margin. The more critical
interruption budget is 100 ms from new speech detection to silence.

Every event carries session ID, turn generation, monotonically increasing
sequence number within its stream, and a deadline. Advancing the generation
cancels future LLM scheduling and TTS, stops playback, and makes late generation-
7 events inert. Only text actually heard by the user is committed to the
visible conversation history; generated-but-unplayed text is recorded as
diagnostic state, not silently treated as spoken.

## 21. Protocol conformance

Define one golden request with a pinned tokenizer and chat template. Test the
non-streamed body and streamed events against the same semantic result. The
suite should assert token counts, finish reason, stop-token exclusion, tool-call
arguments, schema validity, and error shape—not exact wall-clock chunk grouping.

For a slow consumer, cap the per-connection output buffer. When it fills, pause
or cancel that stream without blocking the shared engine output path. After a
disconnect, the request transitions to cancelling, future steps stop, in-flight
output is ignored, and KV references eventually return to the baseline count.

For duplicate request ID `r-17`, choose and document one rule. Atlas rejects a
second live attempt with conflict status; after a completed idempotent request,
it may return the recorded terminal response for a retention window. Tool
execution uses a separate idempotency key, because repeating generation and
repeating an external action are not equivalent.

Run the suite against old and new engine revisions. Any difference is
classified as intended API change, allowed numerical variation, or regression.
“Both returned HTTP 200” is not conformance.

## 22. Benchmark card

A minimally credible card contains:

```text
claim: candidate improves TTFT-qualified goodput for Atlas document traffic
model: exact artifact, tokenizer, precision, context configuration
system: engine commit, container digest, driver/runtime, kernel backends
hardware: device count and memory, CPU/NUMA, links, NIC, power policy
workload: published trace hash; open loop; input/output and prefix distributions
SLO: success, TTFT <= 600 ms, every ITL <= 150 ms, valid output
method: warm-up, cache state, run duration, repetitions, error/cancel policy
outputs: raw request events, server metrics, configuration, analysis revision
quality: task score and structured-output equivalence gate
```

Run baseline and candidate in randomized order at several offered loads. Keep
failed and timed-out requests in the accounting. Report the goodput curve and
confidence intervals, not only the best point.

If the second-day result moves, compare temperature, clock policy, background
traffic, cache warmth, artifact hashes, and workload ordering. The correct
response is to explain or bound the variance. Averaging two different regimes
produces a precise number for no reproducible system.

## 23. Operations runbook

Symptom: p95 TTFT rises from 480 ms to 1.4 s while GPU utilization falls from
72 to 38 percent.

1. Check ingress and tokenizer queue age. If high, route around or scale that
   tier; rollback is removal of temporary capacity after the queue drains.
2. If those queues are normal, inspect engine admission age and reasons. A
   surge in remote-cache waits suggests a cache dependency, not insufficient
   GPU compute.
3. Compare scheduled prefill tokens with graph fallback and compilation events.
   If a new shape is compiling, stop canary traffic or disable that feature for
   the affected route, retaining the old artifact.
4. Check worker readiness and collective progress. Remove a failed group from
   routing before restarting it; draining preserves live state when possible.

At every branch, record the expected confirming signal. “Restart workers” is
not a diagnosis and can destroy the evidence or cached state that explains the
incident.

The failure drill delays KV transfers by 500 ms. Correct behavior is bounded
transfer waiting, conditional recomputation or early rejection, cancellation
cleanup, and recovery without leaked destination blocks. The dashboard must
show stage queue age, transfer count and bytes, timeout reason, recomputation,
and end-to-end goodput.

## 24. Architecture decision

Decision: serve Atlas on self-managed accelerator nodes using four-way tensor-
parallel replicas, token-budget continuous batching, local prefix caching, and
hybrid queue-plus-locality routing. Keep prefill and decode colocated initially;
enable conditional disaggregation only for prompts above a measured transfer
break-even. Use a managed API as an authenticated overflow route for supported
requests, not as an invisible retry.

The choice follows the workload: interactive traffic needs bounded TTFT and
ITL, document traffic benefits from prefix reuse, and the dense model does not
fit on one device at baseline precision. Rejected alternatives are TP8, because
the wider per-layer collective hurts low-concurrency decode; and unconditional
disaggregation, because short prompts do not repay state transfer.

Security boundaries separate public generation, administrative weight/cache
controls, and model artifacts. Tenant cache namespaces default to isolated.
Prompts and KV state follow the same regional deletion policy. Releases pin the
complete model and runtime identity and retain a rollback namespace.

Sensitivity changes the decision. If context length doubles, KV capacity or
lower-precision KV may become binding. If prefix reuse falls below the measured
threshold, locality-aware routing loses value. If peak traffic doubles for less
than worker startup time, a warm pool or overflow capacity is necessary. If the
TTFT objective tightens, separate prefill capacity may become worthwhile. An
ADR is complete only when it names these review triggers and the evidence that
would reopen the decision.
