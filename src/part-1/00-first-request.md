# 0. Your First Inference Request

You have a trained model and enough GPU memory to load it. A GPU is an
accelerator designed to perform many numerical operations in parallel. What
happens when you send the model a prompt—the text that asks it to produce a
response?

Before routing, scheduling theory, or cluster management, there is one request
and one model **replica**—an independently serving copy of the model. This
chapter follows that request from the moment text leaves a user's keyboard to
the moment an answer finishes streaming back. The replica may span several
GPUs, but we treat it as one logical worker and postpone communication between
devices. Every step will reappear, with complications, in later chapters—but
here it is one loop, short enough to hold in your head.

## The model on the wire

Assume the fictional model used throughout this book's exercises: "Atlas," a
dense, decoder-only Transformer with approximately 70 billion **parameters**.
A parameter is one learned number in the model. *Dense* means every generated
token uses the same set of parameters; *decoder-only* means the model produces
a continuation one token at a time.

Atlas stores each parameter in **BF16**, short for *Brain Floating Point 16*.
BF16 is a 16-bit number format commonly used for neural-network weights; each
value occupies two bytes instead of the four bytes used by 32-bit floating
point. Chapter 10 explains the accuracy and performance trade-offs of reduced
precision. For now, its important property is simply its size.

The model has 80 **Transformer layers**—repeated processing blocks containing
attention and feed-forward calculations, the latter being learned matrix
transformations applied to each token position. We will introduce the shape of
its attention state only when we calculate the request's memory use later in
the chapter. The Atlas constants are collected in the reference card near the
end.

**One request through one model worker: the complete path.**

```blockdiag
flowchart LR
    A["Text in"] --> B["Tokenizer"]
    B --> C["Prefill"]
    C --> D["KV cache"]
    D --> E["Decode step"]
    E --> F["Sample"]
    F --> G["Detokenize"]
    G --> H["Text out"]
    F --> D
```

The arrow from sampling back into the KV cache represents **autoregressive
generation**: the model generates one token, appends it to the sequence, and
uses the entire sequence to choose the next token. The loop runs until a stop
condition fires. Everything before prefill is string manipulation; everything
after sampling is string manipulation. The GPU work lives in the middle, and
that middle is where time goes.


The weight footprint is direct arithmetic:

```text
70 billion parameters × 2 bytes = 140 GB
```

That 140 GB must sit in GPU memory before the model can answer anything. It
does not fit on a single 80 GB GPU, so a practical replica divides the weights
across multiple devices. For now, assume that replica is loaded and ready and
follow the request as if the devices formed one worker. Chapter 4 explains how
the devices are connected; Chapter 13 explains how the model is divided among
them.

## Text to tokens

A model does not read text. It reads integers.

A **tokenizer** converts text into the integers a model accepts. It usually
splits a string into **subwords**: pieces that may be a whole common word, part
of a rare word, punctuation, or whitespace. Each piece maps to an integer ID
from a fixed vocabulary. The mapping is deterministic for a given tokenizer
version: the same string always produces the same IDs.

```text
"The quick brown fox" → [464, 4996, 8516, 3143]
```

Four tokens. The vocabulary typically contains 32,000 to 128,000 entries,
covering common words, word fragments, punctuation, and whitespace. Rare words
are split into several tokens; common words are single tokens. The cost of
tokenization is measured in microseconds per token — negligible next to what
comes after.

A **chat template** is a formatting rule that places role markers, system
instructions, and separators around a conversation before tokenization. The
result is a sequence of integer IDs, typically hundreds to tens of thousands
of them, ready for the model.

## What the model is, physically

The model is a stack of Transformer layers. Its parameters are also called
**weights**. Most sit in matrices—large two-dimensional arrays of numbers—that
transform one array of numbers into another. The 80 layers are applied in
sequence: the output of layer 0 feeds into layer 1, layer 1 into layer 2, and
so on through layer 79.

Before the first layer, an **embedding table** converts each token ID into a
vector, a fixed-length array of numbers the model can process. After the last
layer, an output projection converts the final vector into one score for every
token in the vocabulary.

All of these weights sit across the replica's GPU memory, occupying the 140 GB
computed above. They do not change during inference. The model reads them,
repeatedly, every time it processes a token.

## Prefill: processing the prompt

The first phase of inference is **prefill**: processing the entire input before
generating any output. All input tokens pass through every layer together. This
is a large matrix computation in which the model can apply one weight read to
many token positions.

For a 1,000-token prompt, prefill performs roughly 1,000 positions' worth of
arithmetic while reading the 140 GB of weights once. Its **arithmetic
intensity**—the amount of calculation performed per byte moved from memory—is
high, so the GPU's compute units stay busy.

Using the service-time model that recurs throughout this book:

```text
prefill_ms(tokens) = 20 + 0.035 × tokens
```

The 20 ms is fixed overhead: launching **kernels** (small GPU programs),
allocating working memory, and performing initial data movement. The 0.035 ms
per token is the incremental cost once the pipeline is running. For a
1,000-token prompt:

```text
prefill_ms(1000) = 20 + 0.035 × 1000 = 20 + 35 = 55 ms
```

Fifty-five milliseconds from receiving the prompt to completing the first
phase. This is the dominant component of **time to first token** (TTFT) — the
delay the user perceives before the answer starts streaming.

### What prefill creates

Prefill does not just produce output. It creates persistent state.

Inside each Transformer layer, the **attention** operation lets a token retrieve
relevant information from earlier tokens. The current token produces a
**query** vector. Every earlier position has a **key** vector used to measure a
match with that query and a **value** vector containing the information to
retrieve. Several attention heads perform this matching in parallel from
different learned perspectives.

The keys and values must remain in GPU memory because every future decode step
will read them. Keeping them avoids recomputing the entire prompt for every new
token.

This persistent state is the **KV cache** (*key-value cache*). Atlas stores
eight key-value heads per layer, and each key or value contains 128 BF16
numbers. Its size per token across all layers is therefore:

```text
2 (keys and values) × 80 layers × 8 KV heads × 128 dimensions × 2 bytes
= 327,680 bytes
≈ 320 KiB per token
```

For the 1,000-token prompt, the total KV cache created during prefill is:

```text
1,000 tokens × 320 KiB = 320,000 KiB ≈ 312 MiB
```

Here `GB` means a decimal billion bytes. KiB, MiB, and GiB are binary memory
units: each is 1,024 of the preceding unit. Three hundred and twelve MiB of
state, created in 55 milliseconds,
that must remain resident in GPU memory for the entire duration of the
request. This state will grow by 320 KiB with every new token the model
generates.

## Decode: generating the answer

After prefill, the model enters the **decode** phase: the repeated loop that
generates one new token at a time. Each decode step:

1. Reads the model weights (140 GB).
2. Reads all accumulated KV cache entries.
3. Computes one new position's worth of arithmetic.
4. Writes one new KV entry per layer.
5. Produces a vector of logits — one score per vocabulary entry.

The critical difference from prefill is that the model now reads 140 GB of
weights to compute only one new position per active request. The ratio of data
moved to useful computation is poor. Decode is **memory-bandwidth-bound**: its
speed is limited by **memory bandwidth**, the number of bytes the GPU can move
per second, rather than by how quickly it can perform arithmetic.

Each decode step takes approximately 45 ms for a small group of simultaneous
requests in the Atlas cost model. Most of that time is spent streaming weights
from GPU memory through the compute units.

### Sampling: from scores to a token

The **logits** produced by the final layer are raw scores, one per vocabulary
entry. They are not probabilities yet. To select the next token:

1. **Temperature** rescales the logits. Lower values make high-scoring tokens
   more dominant; higher values make alternatives more likely.
2. **Softmax** converts the scores into probabilities that sum to one.
3. Optional filters reduce the choices: **top-k** keeps the `k` most probable
   tokens, while **top-p** keeps the smallest set whose cumulative probability
   reaches a chosen threshold.
4. The server samples from the remaining probabilities. **Greedy decoding**
   instead always chooses the highest-scoring token.

The result is one integer: the ID of the next token. This step is small compared
with running the model, but it carries state. Sampling uses a pseudo-random
number generator: a deterministic sequence controlled by a seed and its
current position. Preserving that state is necessary when a system promises
repeatable output.

### Detokenization: back to text

The selected token ID is converted back to text by the tokenizer's reverse
mapping. The text fragment is sent to the user immediately — this is
streaming. The user sees partial words assemble into sentences while the model
continues generating.

### The loop

The decode loop repeats: read weights, read KV cache, compute one position,
write one KV entry, sample, detokenize, stream. Each iteration takes roughly
45 ms and produces one token. A 200-token response takes about 200 steps,
roughly 9 seconds of decode time.

The loop ends when one of these conditions is met:

- The model emits a special end-of-sequence token.
- The response reaches a caller-specified maximum length.
- The user closes the connection.

The total time for a 1,000-token prompt with a 200-token response is
approximately:

```text
prefill:   55 ms
decode:   200 × 45 ms = 9,000 ms
total:    ~9.1 seconds
```

The user experiences 55 ms of waiting, then roughly 9 seconds of streaming
text at about 22 tokens per second. This is one request, served alone, on
hardware with nothing else to do.

## Following the bytes

**The request owns memory at three different lifetimes.**

```blockdiag
flowchart TB
    A["Server lifetime"] --> B["Model weights"]
    C["Request lifetime"] --> D["Growing KV cache"]
    E["Step lifetime"] --> F["Activations"]
    E --> G["Logits"]
    B --> H["Read every prefill and decode pass"]
    D --> I["Released when the request ends"]
    F --> J["Reused after each step"]
    G --> J
```

**Activations** are the temporary intermediate vectors produced while executing
a layer. They can be reused after the step. Logits are the raw vocabulary
scores just introduced. A summary of where memory goes during this request:

| Object | Size | Lifetime |
| --- | --- | --- |
| Model weights | 140 GB | loaded once, read every step |
| KV cache at end of prefill | 312 MiB | created during prefill, grows during decode |
| KV cache at end of decode | 312 MiB + 200 × 320 KiB ≈ 375 MiB | released when request finishes |
| Activations (per step) | tens of MiB | allocated and freed each step |
| Logits (per step) | vocabulary × 4 bytes ≈ 0.5 MiB | overwritten each step |

The weights dominate the memory budget. The KV cache is the only object that
grows during the request. Activations are temporary workspace. On hardware
with 80 GB of device memory per accelerator, even a single request's KV cache
is a small fraction of capacity — but this changes fast.

## Why one request is misleading

The single-request story above is clean. The GPU does useful work, the user
gets an answer, and memory is comfortable. Now consider what happens when load
increases.

### Ten concurrent requests

Ten users send prompts at approximately the same time, so ten requests are
**concurrent**—in progress together. Each has a 1,000-token input. The model
weights are still 140 GB: they are read, not copied, so ten requests do not
need ten copies. But the KV cache is per-request:

```text
10 requests × 312 MiB = 3.12 GiB after prefill
```

After each request generates 200 tokens of output:

```text
10 × 375 MiB ≈ 3.66 GiB of KV state
```

Still manageable on an 80 GB device. But a benefit appears: **batching**, or
processing a group of requests in one model step. The engine reads the 140 GB
of weights once and applies them to all ten sequences. The same weight traffic
that served one request now serves ten. Each step takes longer—there is more KV
state to read and more arithmetic to perform—but not ten times longer.

This is the fundamental efficiency gain of batched inference: the cost of
reading the weights is shared across sequences.

### One hundred concurrent requests

Push further. One hundred concurrent requests with 1,000-token prompts:

```text
100 × 312 MiB = 30.5 GiB after prefill
```

On an 80 GB device holding part of the 140 GB model—or holding a compressed
version that uses fewer bits per weight—30 GiB of KV state is a significant
fraction of the remaining memory. After each request generates 200 tokens:

```text
100 × 375 MiB = 36.6 GiB
```

And this assumes 1,000-token prompts. A 4,000-token prompt produces 1.22 GiB
of KV state per request. One hundred such requests need 122 GiB — more than
the entire device. The model cannot even hold them all in memory simultaneously.

### The tension

More requests sharing a decode step means better **utilization**—a larger
fraction of the GPU is doing useful work. But more requests also means more KV
cache memory. The engine faces a direct trade-off:

- **Admit more requests**: better throughput, higher GPU utilization, but more
  memory pressure. Eventually the engine must evict cached state or preempt a
  request—pause it and free some of its memory.
- **Admit fewer requests**: lower utilization, wasted bandwidth on weight
  reads that serve too few sequences, but comfortable memory.

This tension—**throughput** (total work completed per second) against memory,
and utilization against **latency** (the time one request waits)—is the reason
the rest of this book exists. Later chapters ask how many requests may run
together, how their growing state should be stored, how work should be divided
across devices, and how the service should behave when demand exceeds capacity.
All are consequences of one fact: sharing the model makes inference more
efficient, while every additional request brings state and delay.

## The numbers, collected

For reference, the constants used above and throughout the book's exercises:

| Quantity | Value | Source |
| --- | --- | --- |
| Parameters | 70 billion | Atlas model definition |
| Weight precision | BF16 (2 bytes) | Atlas baseline |
| Weight footprint | 140 GB | 70B × 2 |
| Layers | 80 | Atlas model definition |
| KV heads per layer | 8 | Atlas model definition |
| Head dimension | 128 | Atlas model definition |
| KV bytes per token | 327,680 (≈ 320 KiB) | 2 × 80 × 8 × 128 × 2 |
| Prefill fixed overhead | 20 ms | service-time model |
| Prefill per-token cost | 0.035 ms | service-time model |
| TTFT for 1,000-token prompt | ≈ 55 ms | 20 + 0.035 × 1000 |
| Decode step time | ≈ 45 ms | at moderate batch size |
| KV for 1,000 tokens | ≈ 312 MiB | 1000 × 320 KiB |

These are planning estimates for a well-tuned system, not promises. Real
measurements on real hardware are always the final authority.

## Try it yourself

The mechanics described above are not hypothetical. You can observe them on
a smaller model with a single GPU.

### Start a server

Install vLLM and launch a server with an 8-billion-parameter instruction-tuned
model (small enough to fit on one consumer GPU with 24 GB of memory):

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto
```

The `--dtype auto` flag lets vLLM choose a numeric format, such as BF16, that
the model and hardware support.
The server loads the model weights into GPU memory and begins listening for
requests on port 8000.

### Send a request

From another terminal, send a prompt and observe the response:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain what a KV cache is in three sentences."}
    ],
    "max_tokens": 128,
    "stream": true
  }' | head -20
```

With `"stream": true`, you will see server-sent events—small messages sent over
one long-lived HTTP response—arrive one at a time. Each carries a token or a
small group of tokens. The gap before the first event is TTFT: prefill plus any
queue wait. The rhythm of subsequent events is the decode cadence.

### What to observe

While the request runs, a few measurements connect to this chapter's content:

1. **GPU memory usage.** Run `nvidia-smi`, NVIDIA's command-line GPU status
   tool, in a third terminal. Note the memory consumed after model loading
   (weights plus runtime overhead) and watch whether it changes during
   generation (KV cache allocation).

2. **Time to first token.** The delay before the first streamed event includes
   tokenization, queueing, prefill, and one decode step. Compare a short prompt
   with a longer one while keeping every other setting fixed.

3. **Token generation speed.** Count the streamed events per second. Compare
   this measured rate with the time of one decode step. Do not expect the Atlas
   estimates to match: this model is smaller and your hardware and software
   revisions determine the actual result.

4. **Concurrent requests.** Open several terminals and send requests
   simultaneously. Watch GPU memory climb (more KV cache) and per-request
   token rate decline (the same bandwidth now serves more sequences). This is
   the throughput-memory tension from the scaling section above, made visible.

The 8B model is not Atlas. Its KV cache is smaller, its decode steps are
faster, and it fits on one device. But the structure is identical: tokenize,
prefill, decode loop, sample, detokenize, stream. Everything observed here
scales, with the same tensions, to the 70B model and beyond.

## Further reading

You do not need these resources to continue to Chapter 1. Use them when one of
the chapter's new concepts deserves a slower or more visual second explanation.

- **Text and tokens:** Hugging Face's [tokenizer introduction](https://huggingface.co/docs/course/en/chapter2/4)
  explains why models consume token IDs and how word and subword tokenization
  differ.
- **Transformers and attention:** Google's [illustrated Transformer
  introduction](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/)
  develops embeddings, attention, and decoder generation visually.
- **BF16 and reduced precision:** NVIDIA's [TensorRT developer
  guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1060/pdf/TensorRT-Developer-Guide.pdf)
  describes FP32, FP16, and BF16 and the trade-off between numerical range,
  precision, memory, and speed.
- **Compute-bound versus memory-bound work:** NVIDIA's [roofline profiling
  guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)
  connects arithmetic intensity, memory bandwidth, and peak computation.
- **Why KV-cache layout matters:** the vLLM [PagedAttention
  paper](https://doi.org/10.1145/3600006.3613165) shows how request state and
  memory fragmentation limit batching in a production inference engine.

## What comes next

This chapter traced one request through one model replica: text to tokens,
tokens through a stack of Transformer layers, persistent state created during
prefill, a bandwidth-bound decode loop, sampling, and text back out. The path
was linear, and the request did not yet compete with other users.

The rest of this book is about what happens when this simple loop must serve
thousands of users simultaneously, across many GPUs, under strict latency and
quality contracts. Chapter 1 introduces the three planes of decisions —
data, control, and management — and the five categories of state that the
service must protect. Chapter 2 defines what "fast" and "good enough" mean
precisely enough to measure. Chapter 3 returns to the model's execution in
full detail, including mixture-of-experts routing, encoder stages, and
diffusion. From there, every chapter zooms into one region of the system that
the single-request loop left simple.

The loop does not change. The engineering is in making it work under
contention.
