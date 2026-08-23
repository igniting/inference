# 19. Diffusion, Image, Video, and World Models

A text-to-image request does not append one pixel at a time. It starts with a
noisy latent representation and repeatedly transforms that latent toward an
image. A text-to-video model may perform the same loop over a large
spatiotemporal tensor. Where language serving grows its output one token per
engine step, a diffusion request holds its *entire* output in memory from the
first step and refines it wholesale — the unit of progress is a full-image
update, and the unit of cost is that update repeated dozens of times.

The serving system still needs batching, parallelism, caching, graphs, and
stage placement—but their meaning changes. Batching keys on shapes that the
caller implies through resolution rather than token count. Caching trades
approximation for compute inside a single request rather than exact reuse
across requests. Graph capture keys on the request's geometry. And the loop's
rigid structure — same shapes every step — makes diffusion, oddly, an easier
target for graphs than language decode.

## Follow the diffusion pipeline

A common pipeline contains four stages:

**Diffusion repeats a denoising stage around persistent latent state.**

```blockdiag
flowchart LR
    P["Prompt"] --> T["Text encoder"]
    T --> C["Conditioning"]
    N["Initial noise"] --> D["Denoising model"]
    C --> D
    D --> S["Step scheduler"]
    S -->|next latent and timestep| D
    S -->|final latent| V["Image or video decoder"]
    V --> O["Media output"]
```


1. a text encoder turns the prompt into conditioning features;
2. a scheduler defines a sequence of noise levels or timesteps;
3. a denoising network, often a diffusion transformer, updates the latent at
   each step;
4. a decoder converts the final latent to pixels or frames.

Safety checks, upscaling, interpolation, or audio generation may add more
stages. The denoiser usually dominates compute because it runs many times: in
the chapter's worked trace, thirty steps at 24 ms contribute 720 of 815 total
milliseconds — 88 percent — so every optimization in this chapter is, one way
or another, an attack on that repeated stage.

Unlike LLM decode, the latent keeps a stable shape during the loop. Every step
advances a full image or video representation: same tensor, same attention
pattern, same kernel launches, step after step. This makes graph capture and
shape-compatible batching attractive in a way language decode is not — a
language engine step changes shape as sequences finish and prefill interleaves,
while a diffusion step is the *same* computation with different numbers in it.

The step is also the natural deadline quantum. A video model producing frames
progressively exposes intermediate latents that are already watchable; a
request's perceived latency tracks when usable frames emerge, not when the
final step lands. Schedulers that treat the whole pipeline as one opaque
request throw that visibility away — but exploiting it has a price worth
stating. Every mid-stream preview must pass through the 55 ms latent decoder,
so a service that renders previews at steps 10 and 20 spends two extra decoder
passes per request: declared as 110 ms of added device work against a
*perceived* latency win of hundreds of milliseconds for the first glimpse.
The exchange can be favorable and still should be made deliberately — cap
preview count, decode at reduced resolution where the VAE allows it, and let
the client's viewport decide whether an early look is worth the capacity.

## Compatible requests can share a batch

Two requests can batch when the active pipeline stage and tensor shapes agree.
For image generation, resolution, latent channels, timestep, guidance mode,
model variant, and backend may all matter. Video adds frame count and temporal
layout. The compatibility set is bigger than language's (sequence length plus
model version) because the denoiser's work items are full tensors rather than
token rows, and because guidance doubles the branch structure — a
classifier-free-guidance request may want both its branches batched together,
or each branch batched across requests, and the two choices conflict.

A serving scheduler can wait briefly for compatible work, merge it, and split
outputs afterward. The waiting window has a computable break-even. Assume two
same-resolution requests arrive close together, and batching two denoisers
raises per-step cost from 24 ms to a declared 31 ms. Run separately, back to
back, the pair completes at 815 + 815 = 1,630 ms; run batched from the start,
both finish near 30 × 31 = 930 ms plus one pipeline's fixed stages. The gap —
roughly 700 ms of machine time — funds a wait: even if the partner arrives
200 ms late, batching beats queuing whenever the wait is shorter than the
partner's entire serialized service time. The window should be sized in those
terms (a fraction of one pipeline duration), not in round milliseconds.
Padding a short video to match a long one may waste too much compute — the
same pad-to-max trap as Chapters 13 and 17, now in the temporal dimension.

Earlier advice often treated video generation as a batch-size-one workload.
That remains reasonable for very large, latency-focused generations, but it is
not a law. SGLang's current
[diffusion documentation](https://docs.sglang.io/docs/sglang-diffusion) includes
compatible-request inference batching for image and video pipelines. The
benefit depends on workload and stage.

### Signatures decide replay

Graph capture meets this compatibility problem head-on, and SGLang's
"breakable CUDA graph" runner — the diffusion-side wrapper in
[`breakable_cuda_graph/runner.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/multimodal_gen/runtime/breakable_cuda_graph/runner.py)
— is a compact study in making replay safe. Its call contract is the one
Chapter 9 argued for, enforced: "Capture is an explicit, idempotent `capture()`
call (driven at warmup) so that serving never triggers a fresh capture." A call
either finds a captured graph for its signature or runs eagerly.

The signature is the interesting part. Tensor leaves key on *shape and dtype*
so values may change per replay; non-tensor constants must join the key because
they are "baked into the captured Python control flow"; mutable objects key on
identity, "to avoid replaying a graph whose eager [execution would differ]."
Miss the key and the runner does not fail silently — it emits a one-shot
diagnostic listing the differing fields, ending in a hint that deserves
framing: "graphs replay only for the exact shapes captured at warmup … the
auto-derived warmup resolution is the model default, which can differ from the
resolutions you actually serve — declare every served resolution explicitly."
A graph-bucket misconfiguration announces itself as *eager fallback with a
warning*, which is why the bucket set must come from served resolutions, not
defaults.

Replay itself is three moves with one subtlety: copy live inputs into the
captured static buffers (`buf.copy_(live, non_blocking=True)`), replay under a
token scope, then **clone the output** — because the other CFG branch "shares
this static output buffer when shapes match," and the caller may still be
holding the conditional branch's result when the unconditional branch replays.
"The clone is one cheap DtoD copy relative to the full DiT" — the price of
static-buffer reuse, charged exactly where aliasing would corrupt results. And
when the structure changes under a matching key — "should not happen" — the
runner falls back to eager "rather than copy mismatched buffers." Every
fallback path chooses correctness over performance without being asked twice.

## Caching trades quality for skipped work

Nearby denoising steps produce similar intermediate features. Cache methods
reuse selected block outputs instead of recomputing the entire network.
[DeepCache](https://arxiv.org/abs/2312.00858) studies reuse of high-level
U-Net features across steps; [TeaCache](https://arxiv.org/abs/2411.19108) uses
timestep-aware differences to decide when cached outputs can be reused in video
diffusion. The two mark the design space's ends:

**Serving optimizations act at different boundaries in the loop.**

```blockdiag
flowchart TB
    R["Request resolution, steps, and conditioning"] --> B["Compatible batching"]
    B --> G["Graph bucket"]
    G --> K["Cross-step cache policy"]
    K --> P["Parallel or staged placement"]
    P --> Q["Latency and visual-quality evaluation"]
```

| Mechanism | Saves | Compatibility condition | Quality risk |
| --- | --- | --- | --- |
| step batching | launch and weight reuse | resolution, step, and model path | none if semantics match |
| graph replay | CPU launch work | captured shape and control flow | none if correct fallback |
| cross-step cache | repeated intermediate compute | sufficiently similar state | approximation drift |
| stage split | independent scaling | transfer cheaper than queue benefit | none, but latency can regress |


| | DeepCache | TeaCache |
| --- | --- | --- |
| Reuses | high-level U-Net features | transformer block outputs |
| Decision signal | fixed interval over the trajectory | accumulated modulated-input distance |
| Granularity | per-block, schedule-set | per-step, signal-set |
| Architecture | U-Net families | DiT families (with a CFG compatibility set) |

Fixed schedules are simple to reason about and audit; signal-driven skips
concentrate the savings where the trajectory is actually redundant. Production
systems increasingly want the second property but must then *trust a runtime
measurement* — which is why the decision's internals matter, not just its
hit rate.

Unlike exact KV reuse for an identical language prefix, diffusion feature
caching is often approximate. Skipping work can change visual quality. The
cache policy needs an error or quality budget, and evaluation should cover
motion, prompt adherence, temporal consistency, and artifacts—not only latency.
The best cache interval may vary over the denoising trajectory — some steps are
more sensitive than others — so a fixed "reuse every N steps" rule is a
baseline, not a policy.

### Inside TeaCache's skip decision

SGLang's integration in
[`cache/teacache.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/multimodal_gen/runtime/cache/teacache.py)
makes the approximate-reuse contract concrete. The signal is the **modulated
input** — the transformer's input after timestep conditioning — and the test is
a relative L1 distance between this step's and the previous step's modulated
inputs: `diff.abs().mean() / prev.abs().mean()`. That raw distance passes
through a per-model polynomial rescale (`np.poly1d(coefficients)`) before it
*accumulates* — the decision variable is the running sum since the last real
compute, not this step's difference alone. When the accumulator crosses
`teacache_thresh`, the step recomputes and the accumulator resets; until then,
the block reuses its cached residual.

Three details carry the systems lessons. First, boundary steps always compute
(`is_boundary_step` forces `should_calc`), because the trajectory's endpoints
shape everything downstream. Second, the check fails open: "Defensive check: if
previous input is not set, force calculation" — an uninitialized cache costs a
compute step, never a corrupted image. Third, classifier-free guidance gets
*separate* positive and negative caches, and models whose CFG structure the
method has not validated "auto-disable TeaCache when CFG is enabled" — an
explicit compatibility gate rather than a silent wrong-answer risk, the same
philosophy as Chapter 10's quantization fallbacks.

There is also a cost the paper view omits: the decision reads
`.cpu().item()` — a device-to-host synchronization *every step*, inside the
loop graph capture would otherwise own. Approximate caching and graph replay
are not free complements; the skip decision buys 24 ms of denoiser work by
spending a sync, and a deployment should measure the exchange rather than
enable both and assume addition. Against the worked trace, skipping ten of
thirty steps saves at most 240 ms — the accumulator design exists precisely to
make those ten the *safe* ten.

The accumulator's shape is also worth internalizing, because it explains the
skip *pattern*. To make it concrete with declared numbers: suppose the
rescaled per-step distances hover around 0.02 in a quiet stretch and 0.06 when
the image is changing fast, with a threshold of 0.15. The quiet stretch skips
seven consecutive steps (7 × 0.02 = 0.14, still under threshold) before one
more forces compute; the busy stretch skips only two. Savings therefore
cluster where the generation is stable — early background refinement, static
regions of a video — and vanish exactly where quality risk is highest. That
self-alignment is the argument for signal-driven policies over fixed
intervals, and it is why hit rate alone is a misleading metric: two policies
with equal skip counts can land their skips in entirely different places on
the trajectory.

## Parallel dimensions follow the latent

Tensor parallelism can split model weights. Sequence parallelism can divide
spatial or temporal tokens. Ring and Ulysses-style attention move or transpose
state so each rank handles part of a long media sequence. Classifier-free
guidance can place conditional and unconditional branches on different ranks.

These dimensions can compose, but each introduces a collective or transfer.
Video shapes are large enough that context-parallel communication can dominate.
Map the logical partition to the fastest links and include decoder or VAE memory
in the plan.

For one pipeline, draw a tensor shape at every stage and step. Mark which
dimension each parallel method splits. If two methods split the same dimension
in incompatible ways, their size arguments may look valid while the execution
plan is not — the composition conflict is invisible in per-method arithmetic
and obvious in the annotated shape table.

### Composing splits on one tensor

The annotated-shape exercise deserves its concrete form. Take the video latent
from the stage-split estimate — frames F, height H, width W, channels C, plus
attention heads D — and watch where each method wants to cut:

| Method | Splits | Communication per step | Wins when |
| --- | --- | --- | --- |
| tensor parallel | C / D (weights) | activations all-reduced each block | single latent too big for one device |
| sequence parallel | spatial tokens H·W | gather at block boundaries | latents wide, weights fit |
| context parallel (ring) | temporal tokens F | ring exchange of key/value each attention | many frames, long attention |
| CFG parallel | branch | none until branch combine | unconditional branch idle otherwise |

Composition works while each claimant takes a *different* axis: CFG on the
branch dimension, ring attention over F, sequence parallelism over H·W, tensor
parallelism over channels. The plan breaks when two methods claim one axis —
Ulysses-style attention partitions *heads*, which collides with a
tensor-parallel scheme that already divided them, leaving some ranks no work
while others carry two roles' communication. And every added split multiplies
the collectives that must include every rank — Chapter 14's participation rule
again: a diffusion step is only as fast as its slowest collective member, and a
composition whose per-step exchange exceeds the 24 ms step budget converts
parallelism into overhead. Price the composition per step, not per method.

## Stages can use different workers

The text encoder, denoiser, and image or video decoder have different compute
and memory profiles. A disaggregated pipeline can scale them independently and
assign different accelerators. Intermediate tensors must move between pools,
and their sizes span orders of magnitude. Assume a common VAE that downsamples
eight times spatially and four times temporally with sixteen latent channels:
a 1080p, 24-frame clip arrives at the decoder boundary as roughly
240 × 135 × 6 × 16 values — about 6 MiB in BF16 — while its text conditioning
is a few hundred kilobytes. The text-embedding boundary is nearly free to
cross; the latent-to-decoder boundary is not, and it lands on the request's
critical path with no compute hiding it.

Stage separation is most attractive when one stage is reused, batched
differently, or strongly imbalanced — a safety checker fanning out over
finished latents, or a text encoder whose queue fills with short prompts while
the denoiser runs seconds per request.

Note how small the pure-transfer term is at image scale. The 6 MiB decoder
latent crosses the declared NVLink-class link in `a + S/b = 20 µs + 6 MiB ÷
450 GB/s ≈ 34 µs` — about a seventh of a percent of one denoising step — and
even a slower storage-class link prices it under a millisecond. The boundary's real cost is
*serialization*: the request now visits two queues and cannot overlap its
decoder wait with anything, so the split pays only through the utilization it
buys, exactly as G's guidance says. Bytes alone almost never settle it.
SGLang's pinned source implements
diffusion parallel groups, compatible batching, graph runners, caching
integrations, and stage disaggregation under
[`multimodal_gen/runtime`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/multimodal_gen/runtime)
— including a dedicated `disaggregation` package with dispatch policies,
roles, and an orchestrator, the E/P/D pattern of Chapter 15 re-derived for
diffusion stages. The code illustrates how text-generation engine ideas can be
adapted without pretending the loops are identical.

## Real-time video keeps a session alive

World models and causal video generators may accept new observations while
producing future frames. The service now owns a long-lived session with
recurrent or attention state. A user may steer the scene, interrupt generation,
or change controls.

The scheduler must meet per-frame deadlines, not merely finish a request. At a
declared 24 fps the per-frame budget is roughly 42 ms — tighter than the
book's ITL SLO — and it recurs forever; a session that misses one deadline has
not queued work, it has dropped frames a human sees. State may need to migrate
when a worker drains. Dropping quality, resolution, frame rate, or lookahead
can be a graceful overload response: cutting lookahead by one chunk refunds
one chunk-period of slack immediately, which is why lookahead is the first
knob to reach for and resolution — which changes graph buckets and batch
compatibility — the last.

Batching live sessions is possible when their frame clocks and shapes align.
One late session should not stall all others, so the batch may need deadlines or
selective dropping.

### Sessions are state machines

The pinned source's
[`realtime/session.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/multimodal_gen/runtime/realtime/session.py)
shows how little machinery this takes when the invariants are strict. Sessions
live in an LRU (`OrderedDict`, `max_sessions=64`); each chunk *attaches* to its
session by id and the cache bounds memory by evicting the least recently
touched. The protocol lives in one integer: the request's `block_idx`. A chunk
arriving with `block_idx > 0` whose session state is missing raises immediately —
"Missing realtime session state" — because a mid-stream chunk after eviction
would otherwise silently fabricate a fresh session and corrupt the stream's
continuity. A chunk with `block_idx == 0` is an epoch boundary: the client is
restarting the stream, so the cache disposes the old state and installs the
new. That is Chapter 17's membership-epoch discipline in miniature — the
stream's first block is a self-declaring generation marker, and everything
after it demands proof the generation is still alive. Eviction disposes state
explicitly (`_dispose_session`), swallowing and logging disposal failures rather
than letting one bad teardown poison the cache — the same defensive shape as
every other state-owner in Part III.

## Worked example: 30 repeated steps

An image pipeline uses 18 ms for text encoding, 30 denoising steps at 24 ms
each, 55 ms for latent decoding, and 22 ms for postprocessing. The timeline
sums as `18 + 720 + 55 + 22 = 815 ms`, of which denoising contributes 720 ms —
88 percent. Every candidate optimization should be priced against that 720.

Decompose the step itself before optimizing it. Assume 3 ms of each 24 is CPU
launch overhead — kernel launches and Python control flow rather than device
math. Thirty steps spend 90 ms on launches, which graph replay can nearly
erase *if* the served resolutions were captured at warmup; the BCG dive's miss
diagnostic is what tells you whether that assumption held in production.
Caching then attacks the remaining ~630 ms of device work, parallelism its
per-step critical path, and distillation the step count itself — three levers
on the same 720, each with a different quality bill.

A cache that safely skips work equivalent to ten steps has a 240 ms upper-bound
saving before lookup and correction — ten skipped steps at 24 ms each — and the
accumulator design from the TeaCache dive is what decides *which* ten. Note the
ceiling's shape: even a perfect policy cannot touch the other 595 ms, so a
deployment hoping for "2× faster" from caching alone needs a different
arithmetic — fewer steps (distillation, better schedulers) or faster steps
(parallelism, graphs), with caching as one term among several.

Doubling resolution changes latent work far more than text encoding. Doubling
each spatial side quadruples latent tokens; attention over them can grow
quadratically, so denoiser step time may rise several-fold while the 18 ms
text stage barely moves. That is geometry, not measurement — the point is that
the *ratio* between stages shifts with shape, so the bottleneck conclusion
must be remeasured per served resolution. Separating stages helps only when
reuse, independent scaling, or better batching repays intermediate transfer and
queueing — the 6 MiB decoder boundary of the previous section is the price
tag on one such split.

## Practice: justify one optimization

Build a per-stage and per-step timeline from the numbers above. Test one caching
policy at two resolutions and one parallel or disaggregated plan. Include graph
buckets, synchronization, intermediate bytes, and every queue.

Report latency and throughput beside declared visual-quality metrics and
blinded samples. State the workload boundary where your optimization loses.
The worked analysis is in
[Appendix G](../appendices/g-worked-solutions.md#19-diffusion-timeline).

The result should explain where time and bytes go, not announce that one
optimization is universally best. Chapter 20 returns to language models in a
different setting: inference embedded inside a training loop.
