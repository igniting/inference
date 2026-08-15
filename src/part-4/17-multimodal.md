# 17. Multimodal and Encoder-Heavy Serving

A user uploads a 20-second video and asks one short question. The language model
may generate only ten tokens, yet the request can be far more expensive than a
long text prompt. The service must fetch and decode the video, sample frames,
resize them, run a vision encoder, and merge the resulting features into the
language-model input before the first output token appears.

If you measure only language decode, you miss most of the request.

## Media begins as untrusted bytes

Images, audio, video, and documents arrive in formats optimized for storage and
transport. Their decoded representation can be much larger. A small compressed
video may expand into hundreds of frames. A document may contain many high-
resolution pages.

The frontend should validate type, byte size, dimensions, duration, frame
count, and decompression limits before expensive work begins. Fetching remote
media needs timeouts, address restrictions, and a policy for redirects. Media
parsers and codecs belong inside the service's security boundary.

Preprocessing then converts the input into the exact form expected by the model:
resizing, normalization, frame sampling, audio resampling, or document layout
processing. These choices affect model output, so processor version is part of
the execution identity and cache key.

## The encoder is a separate workload

A vision or audio encoder usually processes many positions in parallel. Its
shapes depend on resolution, patch count, frame count, or audio duration. It may
be compute-heavy while language decode is memory-bound.

The output is a tensor of media features. A projection layer maps those features
to the language model's representation, and placeholder positions tell the
language model where they belong.

```text
media -> decode and normalize -> encoder -> projected features
                                                   |
text -> template -> tokenizer ---------------------+
                                                   v
                                     language prefill -> decode
```

The placeholder and feature lengths must agree. Truncating a text prompt can
accidentally remove media positions. Batching requests with different numbers
of images or frames requires ragged metadata. These are correctness concerns,
not only tensor-shape concerns.

## Batch compatible encoder work

Encoders benefit from batching, but compatible shapes matter. Padding every
image to the largest resolution in a batch may waste more compute than batching
saves. Bucketing by resolution, aspect ratio, frame count, or audio length can
improve efficiency while adding queueing delay.

The right batching window depends on the endpoint. An offline document-indexing
job can wait for a full batch. An interactive visual question should not wait
long for another image of the same size.

Measure preprocessing and encoder queueing separately. A busy CPU decoder can
starve an otherwise idle accelerator. Moving some transforms to the GPU can
help, but it also competes with model execution and may create extra copies.

## Encoder outputs can be reused

Users often ask several questions about the same image, document, or video.
Reusing encoder features avoids repeated media work. The cache key should
include content, preprocessing configuration, encoder and projection weights,
precision, and any model-specific placeholder layout.

Caching processed bytes alone saves decoding. Caching encoder outputs saves
more compute and consumes more space. Caching language KV state after the first
question can save still more, but may be tied to the exact conversation
template. These are distinct cache layers with different reuse scopes.

Treat privacy carefully. Encoder features can reveal information about the
original media. Apply the same tenant, retention, and deletion policy used for
the source content.

## Separate encoder, prefill, and decode when it pays

An encoder-heavy service can place media encoders in their own worker pool. The
pool batches media efficiently and sends features to language-prefill workers.
Decode runs in a third pool. This E/P/D topology allows independent scaling and
hardware selection.

The boundary also adds a queue and a transfer. Short, uncached media may be
faster on a colocated worker. Long videos or repeated questions may benefit from
separation. Route conditionally based on estimated encoder time, feature size,
cache state, and queueing.

vLLM's official
[disaggregated-encoder documentation](https://docs.vllm.ai/en/stable/features/disagg_encoder/)
describes independent scaling, TTFT isolation, and cross-process reuse as
motivations. The pinned source contains encoder-cache and transfer integration
in the scheduler and under
[`distributed/ec_transfer`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/distributed/ec_transfer).
SGLang's encode server, receiver, and multimodal scheduling paths live under
[`srt/disaggregation`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/disaggregation)
and its managers package.

## Not every encoder leads to generation

Embedding, classification, reranking, and reward endpoints usually produce a
complete output after one model pass. Their main serving questions are dynamic
batching, padding, pooling, output normalization, and latency limits.

Reranking is a useful example. One query may be paired with hundreds of
documents. The service can flatten pairs into a batch, but must restore scores
to the correct request and document order. Large requests can monopolize the
batch unless scheduling limits work per request.

Embedding APIs need a precise normalization and truncation contract. Reward
models may return one scalar per candidate or token-level values. Using a
generation-oriented output path without defining these semantics creates subtle
compatibility errors.

## Profile the path to first output

Take one representative multimodal request and place timestamps at byte
receipt, media fetch, decode, preprocessing, encoder queue, encoder execution,
feature transfer, language queue, prefill, and first output.

Repeat with the same media and a different question. Then compare no cache,
processed-media cache, encoder-output cache, and full language-prefix reuse.
Measure latency saved per byte retained and verify output equivalence.

The exercise makes the chapter's main point visible: multimodal inference is a
pipeline of independently schedulable work. Chapter 18 examines another
pipeline whose expensive stage repeats over time—diffusion generation.
