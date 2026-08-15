# 11. Speculative and Constrained Decoding

Autoregressive generation normally pays for one large-model step per token. If
a response contains 200 tokens, the target model runs at least 200 serial
iterations.

Speculative decoding asks whether a cheaper method can guess several future
tokens and let the target model verify them in parallel. When the guesses are
good and cheap, one target step advances the sequence by more than one token.

## Draft, verify, accept

The classic method uses a small draft model. The draft proposes a run of tokens.
The target model evaluates the proposed positions together. An acceptance rule
keeps a valid prefix of the proposal and corrects the first rejected position.

```text
draft:   [the] [device] [is] [ready]
target:    ✓      ✓      ✗
result:  accept two tokens, correct the third, discard the rest
```

With the appropriate rejection-sampling rule, speculation preserves the target
model's output distribution. The foundational
[speculative decoding paper](https://arxiv.org/abs/2211.17192) describes this
exact approach.

The speedup depends on more than acceptance rate. Drafting consumes compute.
Verification uses larger target batches and extra KV slots. Rejected tokens
create wasted work. The scheduler and graph runner must support several
proposal lengths.

A simple mental model is:

```text
benefit = target steps avoided
cost    = drafting + larger verification + rejection waste + coordination
```

Speculation helps only when the saved target work exceeds the full cost.

## Several ways to propose tokens

A separate draft model is not the only source of guesses.

Multi-token prediction heads and EAGLE-style drafters use features from the
target model to propose future tokens. Native MTP layers are trained into some
model architectures. N-gram and suffix methods search the prompt or recent
history for repeated continuations. They add little model compute and work well
on repetitive text, but fail when the continuation is novel.

Tree methods propose several branches so the target verifies multiple possible
continuations. They may improve the chance of advancing and also enlarge the
verification workload.

At the pinned revisions, vLLM contains draft-model, EAGLE, MTP, n-gram, suffix,
DFlash, and dynamic verification paths under
[`vllm/v1/spec_decode`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/spec_decode).
SGLang's corresponding implementations live under
[`srt/speculative`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/speculative).
Support and compatibility vary by model and backend; the source trees illustrate
the design space rather than one default recipe.

## Proposal length should adapt

A fixed proposal length is easy to graph and schedule. It is wasteful when
acceptance changes.

At low batch size, target decode may be memory-bound, making a larger
verification batch relatively cheap. Under high concurrency, the target is
already efficient and draft work competes with useful requests. Some prompts
are predictable; others are not. Acceptance also changes during one response.

An adaptive policy can use recent acceptance, batch size, draft confidence, or
an estimated cost model to choose the number of proposed tokens—or disable
speculation entirely. The policy must be stable enough to avoid frequent graph
misses and schedule churn.

Measure accepted tokens per target step, not acceptance percentage alone. A
method accepting 80 percent of two proposals advances less than one accepting
60 percent of eight if their costs are comparable.

## Speculation changes memory and scheduling

The target needs temporary positions for proposed tokens. Accepted positions
become normal sequence state; rejected positions must not remain visible.
Chunked prefill creates boundaries where a drafter may need additional
lookahead. Asynchronous scheduling can prepare the next step before acceptance
is known, so it must reserve conservatively and repair state afterward.

Parallel execution adds synchronization. Every rank must agree on accepted
lengths and state mappings. Disaggregated decode adds another question: where
does the drafter run, and which state crosses the network?

This is why speculative decoding is a serving algorithm, not a wrapper around
model calls.

## Constrained generation solves a different problem

Many applications need output that follows JSON Schema, a regular expression,
or a grammar. A constrained decoder tracks the grammar state for each sequence
and masks tokens that would make a valid completion impossible.

The grammar may be compiled into a finite-state representation before decode.
At each step, the engine determines allowed tokens and applies a mask to the
logits. Efficient implementations cache grammar states and use bitmasks on the
GPU.

Tool and reasoning parsers add streaming semantics. A tool-call argument may be
incomplete for many tokens before it becomes valid JSON. The API must not emit a
final event too early, and cancellation must clean up parser state.

Constrained generation and speculation interact. A draft token forbidden by
the current grammar cannot be accepted. Masking may change proposal quality,
and verification must use the same constraint state as ordinary decode.

The pinned repositories expose these concerns in vLLM's
[`structured_output`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/structured_output)
and SGLang's
[`constrained`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/constrained)
packages.

## Find the workload where speculation loses

Benchmark ordinary decode and at least two proposal strategies on predictable,
unpredictable, short-output, and high-concurrency workloads. Record draft time,
verification time, accepted tokens per target step, extra memory, graph
dispatch, TTFT, ITL, and output equivalence.

Then search deliberately for the losing case. A useful speculative system knows
when not to speculate. Reporting only the prompt class with high acceptance
turns an adaptive resource decision into a misleading universal claim.

We have now followed a request through one engine, from allocation to kernels
and decoding. Part III expands the same ideas across multiple accelerators and
machines.
