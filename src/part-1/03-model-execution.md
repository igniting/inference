# 3. How Generative Models Execute

When a language model writes a sentence, it does not plan the whole sentence
and reveal it one word at a time. It repeatedly predicts what should come next.
That simple loop shapes almost every part of an LLM server.

An engine does not schedule an abstract “model.” It schedules operations with
particular tensor shapes, dependencies, and state. To understand the engine, we
first need to understand the work the model creates.

The original Transformer architecture is described in
[Attention Is All You Need](https://arxiv.org/abs/1706.03762). Serving-oriented
architectures have changed many details since then, so this chapter emphasizes
the execution properties an engine must discover rather than one fixed block
diagram.

## The autoregressive loop

A decoder-only language model begins with token IDs. It converts them to
vectors, passes those vectors through a stack of transformer blocks, and
produces a score for every possible next token. Sampling rules turn the scores
into one selected token. The selected token is appended to the sequence, and
the process repeats.

```text
tokens -> transformer blocks -> logits -> sampling -> next token
   ^                                                 |
   +-------------------------------------------------+
```

The next iteration depends on the token selected in the previous one. This is
the serial dependency behind decode latency. A server can process many
sequences together, but one sequence cannot generate its tenth new token before
it knows the ninth.

## One model, two kinds of work

The first pass over the prompt is called **prefill**. The model processes many
input positions at once and creates the attention state needed later. Large
matrix operations during prefill tend to use the accelerator's compute units
well.

After prefill, the model enters **decode**. Each active sequence usually adds
one position per step. At a small batch size, the GPU repeatedly reads a large
set of weights to do relatively little arithmetic. Decode is therefore often
limited by memory traffic or launch overhead.

Batching more sequences lets the same weight read serve more work. That raises
throughput, but a request may wait longer for its place in the batch. The
scheduler spends much of its life balancing this exchange between hardware
efficiency and user latency.

Prefill and decode use the same weights, yet behave like different workloads.
Later chapters will use that fact to motivate chunked prefill, separate graph
shapes, phase-specific parallelism, and disaggregated serving.

## Attention remembers the past

Inside a transformer block, attention lets each position combine information
from earlier positions. Recomputing the entire prompt for every new token would
be wasteful. Instead, the model stores the keys and values created for previous
positions. This persistent state is the **KV cache**.

For a conventional attention layout, a rough size estimate for one sequence is:

```text
KV bytes = 2 * layers * tokens * KV heads * head dimension * bytes per value
```

The factor of two accounts for keys and values. The total grows with sequence
length and can become much larger than the temporary activation memory of one
decode step.

Modern architectures change the formula. Grouped-query and multi-query
attention use fewer KV heads. Multi-head latent attention stores compressed
latent state and separate positional components. Sliding-window attention only
needs a recent region. Recurrent and state-space layers can keep fixed-size
state instead of one entry per token.

This means that a modern engine may manage several kinds of persistent state in
the same model. Calling all of it “the KV cache” is convenient, but assuming it
has one shape or one retention rule is not.

## Attention patterns change what can be reused

Full causal attention allows a new token to attend to every earlier token.
Other patterns limit the receptive field.

A sliding-window layer attends only to recent positions. A local or
block-sparse layer follows a fixed pattern. Cross-attention reads state produced
by an encoder. Some architectures share state between layers or summarize old
positions into recurrent state.

The model runner must pass the correct positions, mask, page table, and
layer-specific metadata to the attention implementation. Selecting an
attention backend is therefore a correctness decision before it becomes a
performance decision.

## Mixture-of-experts models add routing

A dense feed-forward layer applies the same parameters to every token. A
mixture-of-experts, or MoE, layer contains many feed-forward networks called
experts. A router selects a small number of experts for each token.

This conditional computation allows the model to contain many parameters
without using all of them for every token. It also introduces irregular work.
Tokens must be grouped by expert so the accelerator can run efficient matrix
multiplications. If experts live on different GPUs, token representations must
move to the selected owners and return afterward.

The busiest expert determines when the step finishes. A model with balanced
average routing can still have a hot expert for a particular workload. Serving
MoE models is therefore as much a placement and communication problem as a
matrix-multiplication problem.

## Sampling has state too

The model's logits are not always sampled directly. Temperature, top-k and
top-p filters, repetition penalties, token bans, grammars, and custom logit
processors can all change the distribution. Random sampling owns a generator
state. Structured generation owns a parser or finite-state machine for each
sequence.

Greedy selection chooses the highest-scoring token. It is deterministic only if
the logits and tie handling are identical. Different batch shapes, reduction
orders, kernels, precisions, collectives, or cache paths can slightly change the
logits. Temperature zero does not by itself guarantee identical output across
executions. Later chapters will separate deterministic selection from
deterministic numerical execution.

## Models with encoders

Now consider a user who attaches an image to a question. The service may decode
the image, resize and normalize it, run a vision encoder, project the resulting
features, and insert them into the language-model input. Only then do language
prefill and decode begin.

```text
image bytes -> preprocessing -> vision encoder -> media features
                                                    |
text ----------> tokenization ----------------------+
                                                    v
                                      language prefill -> decode
```

For a large image or video, the encoder can dominate time to first token. Its
output may be reusable when the user asks several questions about the same
media. Resolution and frame count also create dynamic shapes. The encoder is a
serving stage with its own batching, caching, and placement decisions—not a
minor preprocessing detail.

Embedding, reranking, classification, and reward models often have no decode
loop at all. They batch complete inputs and produce complete outputs. Engines
that support these tasks need schedules and output paths suited to them.

## Diffusion follows a different loop

An image-generation pipeline commonly contains a text encoder, a denoising
network, a scheduler that chooses noise levels, and a decoder that turns a
latent representation into pixels. The denoising network runs many times.

Unlike an autoregressive sequence, a diffusion request often advances all
spatial positions together. Its latent state may keep a stable shape across
steps. Requests can share a batch when their resolution, step, conditioning,
and backend requirements are compatible. Some systems cache repeated work
between nearby denoising steps.

The lesson is broader than diffusion: an inference engine should model stages,
dependencies, and state rather than assume that every request emits one token
per iteration.

## From model topology to serving topology

The **model topology** describes what depends on what: layers, experts,
encoders, attention state, and iterative stages. The **serving topology** maps
that work onto devices and processes.

The same model can be replicated in full, split across devices by tensor or
layer, distributed by expert, or separated into encoder, prefill, and decode
pools. All may be legal. The workload, hardware links, memory capacity, and SLO
decide which is useful.

Before choosing an engine configuration, take one dense decoder, one MoE
decoder, and one hybrid model. Estimate parameter bytes, persistent state per
token, prefill and decode shapes, conditional communication, and separately
placeable stages. This exercise turns model names into the work a serving
system must actually perform.

The next chapter maps that work onto real memory and interconnects.
