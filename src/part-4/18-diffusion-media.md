# 18. Diffusion, Image, Video, and World Models

A text-to-image request does not append one pixel at a time. It starts with a
noisy latent representation and repeatedly transforms that latent toward an
image. A text-to-video model may perform the same loop over a large
spatiotemporal tensor.

The serving system still needs batching, parallelism, caching, graphs, and
stage placement—but their meaning changes.

## Follow the diffusion pipeline

A common pipeline contains four stages:

1. a text encoder turns the prompt into conditioning features;
2. a scheduler defines a sequence of noise levels or timesteps;
3. a denoising network, often a diffusion transformer, updates the latent at
   each step;
4. a decoder converts the final latent to pixels or frames.

Safety checks, upscaling, interpolation, or audio generation may add more
stages. The denoiser usually dominates compute because it runs many times.

Unlike LLM decode, the latent often keeps a stable shape during the loop. Every
step advances a full image or video representation. This makes graph capture
and shape-compatible batching attractive.

## Compatible requests can share a batch

Two requests can batch when the active pipeline stage and tensor shapes agree.
For image generation, resolution, latent channels, timestep, guidance mode,
model variant, and backend may all matter. Video adds frame count and temporal
layout.

A serving scheduler can wait briefly for compatible work, merge it, and split
outputs afterward. The waiting window must respect user latency. Padding a
short video to match a long one may waste too much compute.

Earlier advice often treated video generation as a batch-size-one workload.
That remains reasonable for very large, latency-focused generations, but it is
not a law. SGLang's current
[diffusion documentation](https://docs.sglang.io/docs/sglang-diffusion) includes
compatible-request inference batching for image and video pipelines. The
benefit depends on workload and stage.

## Caching trades quality for skipped work

Nearby denoising steps can produce similar intermediate features. Cache methods
reuse selected block outputs instead of recomputing the entire network.

[DeepCache](https://arxiv.org/abs/2312.00858) studies reuse of high-level U-Net
features across steps. [TeaCache](https://arxiv.org/abs/2411.19108) uses
timestep-aware differences to decide when cached outputs can be reused in video
diffusion.

Unlike exact KV reuse for an identical language prefix, diffusion feature
caching is often approximate. Skipping work can change visual quality. The
cache policy needs an error or quality budget, and evaluation should cover
motion, prompt adherence, temporal consistency, and artifacts—not only latency.

The best cache interval may vary over the denoising trajectory. Some steps are
more sensitive than others. A fixed “reuse every N steps” rule is a baseline.

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
plan is not.

## Stages can use different workers

The text encoder, denoiser, and image or video decoder have different compute
and memory profiles. A disaggregated pipeline can scale them independently and
assign different accelerators.

Intermediate tensors must move between pools. Text embeddings are usually
small compared with video latents; the decoder boundary may carry a large
tensor. Stage separation is most attractive when one stage is reused, batched
differently, or strongly imbalanced.

SGLang's pinned source implements diffusion parallel groups, compatible
batching, graph runners, caching integrations, and stage disaggregation under
[`multimodal_gen/runtime`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/multimodal_gen/runtime).
The code illustrates how text-generation engine ideas can be adapted without
pretending the loops are identical.

## Real-time video keeps a session alive

World models and causal video generators may accept new observations while
producing future frames. The service now owns a long-lived session with
recurrent or attention state. A user may steer the scene, interrupt generation,
or change controls.

The scheduler must meet per-frame deadlines, not merely finish a request. State
may need to migrate when a worker drains. Dropping quality, resolution, frame
rate, or lookahead can be a graceful overload response.

Batching live sessions is possible when their frame clocks and shapes align.
One late session should not stall all others, so the batch may need deadlines or
selective dropping.

## Build a stage timeline

Profile one image or video pipeline. Record text encoding, each denoising step,
cache hits, communication, decoding, and postprocessing. Repeat at several
resolutions and batch sizes.

Next, test one caching policy and one parallel plan. Report visual-quality
metrics and blinded samples beside latency and throughput. Finally, estimate a
disaggregated placement and include intermediate transfer time.

The result should explain where time and bytes go, not announce that one
optimization is universally best. Chapter 19 returns to language models in a
different setting: inference embedded inside a training loop.
