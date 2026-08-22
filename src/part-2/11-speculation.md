# 11. Speculative and Constrained Decoding

Autoregressive generation normally pays for one large-model step per token. If
a response contains 200 tokens, the target model runs at least 200 serial
steps. That serialization is not an accident — each token conditions on
the last — but it is also not sacred: nothing prevents the model from
*checking* several guesses in one pass, because verification of a fixed
candidate sequence is parallel while generation of it is not.

Speculative decoding asks whether a cheaper method can guess several future
tokens and let the target model verify them in parallel. When the guesses are
good and cheap, one target step advances the sequence by more than one token.
When they are bad, the engine pays for the guesswork and the verification and
advances no faster than before. The whole discipline is in knowing which
regime the traffic is in — and in the machinery that keeps a rejected guess
from corrupting committed state.

## Visual map

**Speculative decoding proposes several tokens but commits only verified work.**

```blockdiag
flowchart LR
    C["Current accepted prefix"] --> D["Draft proposes tokens"]
    D --> V["Target verifies proposal"]
    V --> A{"Accepted prefix length"}
    A --> K["Commit accepted tokens"]
    K --> C
    A --> F["Sample correction at first rejection"]
    F --> C
```

**A speculation round is a strip of drafts with a rollback point at the first
rejection.**

```blockdiag
flowchart LR
    P["Committed prefix"] --> T1["Draft t+1"] --> T2["Draft t+2"] --> T3["Draft t+3"] --> V{"Verify all positions in one target step"}
    V -->|"all accepted"| C["Commit through t+3"]
    V -->|"reject at t+2"| B["Commit through t+1;<br/>t+2 rolls back and resamples"]
    C --> P
    B --> P
```

**Constrained decoding adds parser state to every sampling step.**

```blockdiag
flowchart LR
    L["Model logits"] --> M["Grammar or schema mask"]
    P["Parser state"] --> M
    M --> S["Sample legal token"]
    S --> U["Update parser state"]
    U --> P
    S --> O["Output stream"]
```

| Mechanism | Added state | Expected benefit | Common losing case |
| --- | --- | --- | --- |
| draft model | draft weights and KV | several accepted tokens per target step | low acceptance or memory pressure |
| n-gram proposal | prefix index | cheap repeated-text proposals | novel text |
| grammar mask | parser or automaton state | valid structured output | complex masks or unsupported batching |
| lookahead slots | reserved cache capacity | stable proposal execution | reduced normal concurrency |

## Draft, verify, accept

The classic method uses a small draft model. The draft proposes a run of tokens.
The target model evaluates the proposed positions together. An acceptance rule
keeps a valid prefix of the proposal and corrects the first rejected position.

Where the draft runs is itself a placement decision. On the same device it
steals compute from the target between steps; on a separate device it adds a
transfer of features or tokens per round; on the CPU it avoids device
contention but rarely keeps up with a multi-millisecond step. The draft also
owns KV state of its own — it must attend over the same growing context to
propose well — so the table's "draft weights and KV" line is a second cache
growing in lockstep with the first, and the memory accounting of the next
section has to count it.

```text
draft:   [the] [device] [is] [ready]
target:    ✓      ✓      ✗
result:  accept two tokens, correct the third, discard the rest
```

With the appropriate rejection-sampling rule, speculation preserves the target
model's output distribution. The intuition behind the rule: a draft token is
accepted in proportion to how much the target agrees with the draft, and a
rejected position is resampled from the target's own distribution renormalized
over what remains — so the committed stream is statistically indistinguishable
from pure target sampling. The foundational
[speculative decoding paper](https://arxiv.org/abs/2211.17192) describes this
exact approach. Distribution preservation is the property that makes the
technique a *serving* optimization rather than an approximation: quality is
not traded for speed, only latency and throughput are reshaped.

Two invented probabilities make the rule inspectable. Suppose the draft's top
choice is "device" with draft probability 0.8 while the target gives it 0.6:
acceptance probability is min(1, 0.6/0.8) = 0.75, so three times in four the
target keeps the draft's guess. On rejection, the token is resampled from the
target's distribution *minus* the mass already spent on the draft choice,
renormalized — which is exactly why the committed stream has the same
distribution as if the target had sampled alone, even though the draft chose
what to offer. The agreement rate sets both the acceptance probability and how
much correction mass remains, which is why drafter-target agreement, not draft
quality alone, is what an acceptance metric should surface.

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

### What verification actually costs

The verification batch is the hidden price, and Chapter 4's arithmetic prices
it. A decode step's arithmetic intensity is roughly the batch size, and the
crossover where decode stops being memory-bound sits near intensity 333 for
the Atlas hardware class. With three speculative tokens, every decode slot
carries 1 + 3 = 4 tokens — the same stride Chapter 9 found in
`uniform_decode_query_len`. At batch 8 the verification step runs at intensity
about 32, still far below crossover, so the extra tokens ride along almost
free: memory-bound decode was reading the weights once regardless. At batch
128 the ordinary step already runs near intensity 128, and verification
quadruples it to about 512 — past crossover, where the target is no longer
idle and every speculative token competes with real work. The same proposal
length that is free for an interactive tier is a tax under high concurrency.

Memory carries a quieter version of the same bet. Before acceptance is known,
each active sequence needs reserved positions for its proposal — the
lookahead slots in the visual map's table. At 320 KiB per token, four
lookahead positions cost about 1.3 MiB per sequence: trivial per request, but
it is capacity that serves no user when acceptance is low, and the graph
runner needs buckets covering the padded proposal shapes on top. Speculation's
costs arrive as capacity, scheduling, and shape coverage before they arrive as
milliseconds.

## Several ways to propose tokens

A separate draft model is not the only source of guesses.

Multi-token prediction heads and EAGLE-style drafters use features from the
target model to propose future tokens. The advantage is structural: an
EAGLE-style drafter conditions on the target's own hidden state, so its first
guess already benefits from everything the target computed this step, and its
drafts correlate with target behavior in a way an independent small model
cannot match. Native MTP layers are trained into some
model architectures — the head ships with the checkpoint, and its acceptance
profile is a property of the model rather than of a separately chosen draft.
N-gram and suffix methods search the prompt or recent
history for repeated continuations. They add little model compute and work well
on repetitive text, but fail when the continuation is novel.

Tree methods propose several branches so the target verifies multiple possible
continuations. They may improve the chance of advancing and also enlarge the
verification workload — and the enlargement is multiplicative, not additive.
A chain of three drafts verifies three positions; a tree with two branches at
each of three depths verifies up to seven nodes, and the KV lookahead
reservation grows with the node count, not the path length. The attention
kernel must mask across the branch structure, so tree verification also
narrows kernel compatibility. Trees pay off where single chains stall: text
that is locally unpredictable but globally repetitive, where several
continuations are plausible and committing to one wastes the round.

At the pinned revisions, vLLM contains draft-model, EAGLE, MTP, n-gram, suffix,
DFlash, and dynamic verification paths under
[`vllm/v1/spec_decode`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/spec_decode).
SGLang's corresponding implementations live under
[`srt/speculative`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/speculative).
Support and compatibility vary by model and backend; the source trees illustrate
the design space rather than one default recipe.

### Guided reading: two proposers, two philosophies

vLLM's
[`ngram_proposer.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/spec_decode/ngram_proposer.py)
is the zero-model end of the design space. Its `load_model` method is literally
"No model to load": the proposer searches each request's own tokens — prompt
plus generated history — for the longest suffix that appeared earlier, then
proposes the tokens that followed it. The search is a Knuth-Morris-Pratt
failure-function scan over the reversed token array, compiled through numba,
with the match length capped at `max_ngram` "to save memory" and a subtle
tie-break the comment explains: on equal-length matches it keeps "the earliest
position in the original tokens," preferring the *first* time history
repeated rather than the most recent. The systems details are as instructive
as the algorithm. The constructor runs one throwaway proposal to trigger JIT
compilation — warm-up discipline, same as Chapter 9's captures. The thread
budget is deliberately tiny: capped at one thread, then divided by TP size
"to ensure each tensor parallel rank has some threads since all ranks will run
this," because — the comment continues — "other components like frontend
(incl tokenization) and Structured Outputs also use multiple threads." A
CPU-side proposer competes for cores with everything else on the host, and
every rank runs it redundantly to stay in lockstep.

SGLang's `adaptive_spec_params.py` is the closed-loop end: it treats proposal
length as a control variable. `DEFAULT_ADAPTIVE_CONFIG` assigns each batch
size its own candidate set — steps `[1, 3, 7]` at batch 1, `[0, 1, 3]` at
batch 8, `[0, 1]` at batch 32, and `[0]` at batch 64. Read that last entry
carefully: by default, speculation *disables itself* under high concurrency,
exactly the regime the intensity arithmetic above predicted. Each slot tracks
an EMA of accepted draft length and follows the docstring's rule —
`target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)` —
probing one step beyond observed acceptance, updating only every five batches
after ten warm-up batches so the controller does not chase noise. Hysteresis
is asymmetric (dropping is easier than rising), a zero-step interval is
treated as a probe state that restarts from the smallest positive candidate,
and the EMA ceiling "only caps downward — never blocks step-ups, so the
system can explore higher steps and let the EMA catch up." Routing closes the
loop with Chapter 9: `_route(batch_size)` pads the batch to its CUDA-graph
size and picks the nearest configured slot, so the controller only ever
selects step counts the graph buckets can execute. And
`adaptive_unsupported_reason` enumerates what the controller cannot coexist
with — DP attention ("adaptive tier decisions are not synchronized across DP
ranks"), two-batch overlap ("adaptive state swap would discard the
TboAttnBackend wrapper") — a list that reads like a map of Part III. A
runtime-tuned knob is still a knob with preconditions. Read together, the two
files bracket the design space: one buys proposals with CPU cycles and an
old algorithm, the other with a control loop over the serving systems this
whole part has built.

## Proposal length should adapt

A fixed proposal length is easy to graph and schedule. It is wasteful when
acceptance changes.

At low batch size, target decode may be memory-bound, making a larger
verification batch relatively cheap. Under high concurrency, the target is
already efficient and draft work competes with useful requests. Some prompts
are predictable; others are not. Acceptance also changes during one response —
code and boilerplate accept well; the sentence that introduces a new idea
accepts poorly.

An adaptive policy can use recent acceptance, batch size, draft confidence, or
an estimated cost model to choose the number of proposed tokens—or disable
speculation entirely. The policy must be stable enough to avoid frequent graph
misses and schedule churn, which is why the controller above smooths with an
EMA, hysteresis, and an update interval instead of reacting to every batch.

Measure accepted tokens per target step, not acceptance percentage alone. A
method accepting 80 percent of two proposals advances less than one accepting
60 percent of eight if their costs are comparable.

The expectation arithmetic shows why proposal length hits diminishing returns
fast. If each draft position independently survives with probability p, a
chain of k proposals advances 1 + p + p² + … + pᵏ tokens on average — the 1
is the correction token every round earns. At p = 0.8 and k = 3 that is
1 + 0.8 + 0.64 + 0.51 ≈ 2.95 tokens per round. Stretching to k = 6, each new
position buys less than the last — the fourth draft is worth p⁴ ≈ 0.41
tokens, the fifth 0.33, the sixth 0.26 — while every one of them costs a full
verification slot and lookahead reservation. The geometric tail means long
chains only pay when acceptance is very high: at p = 0.95 the same stretch
from three to six adds about 2.3 tokens per round, which is why predictable
traffic tolerates aggressive proposal lengths that would bankrupt
unpredictable traffic.

## Speculation changes memory and scheduling

The target needs temporary positions for proposed tokens. Accepted positions
become normal sequence state; rejected positions must not remain visible.
Chunked prefill creates boundaries where a drafter may need additional
lookahead. Asynchronous scheduling can prepare the next step before acceptance
is known, so it must reserve conservatively and repair state afterward —
Chapter 6's overlap machinery now settles bets, not just samples.

Memory pressure turns the lookahead reservation into a preemption input,
using Chapter 6's distinctions exactly. A preempted sequence that is swapped
moves its reserved lookahead blocks to the host too — paying transfer cost on
state that may be discarded unverified when the sequence returns. A preempted
sequence that is recomputed pays 0.06 ms per token for its whole accepted
context under Atlas's constants — including tokens that speculation produced
nearly for free, which are now ordinary committed history that must be earned
back the slow way. An admission controller that counts only committed context
will find sequences heavier than they look; the lookahead slots are part of
the sequence's true footprint.

Parallel execution adds synchronization. Every rank must agree on accepted
lengths and state mappings; a single rank that kept a rejected token would
fork its KV cache from its peers' and corrupt every subsequent collective.
Disaggregated decode adds another question: where
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

The mask's size is worth pricing once. A 128k vocabulary needs 128k bits of
legality per position — 16 KiB as a packed bitmask, about three percent of
the 512 KB logits row Chapter 5 counted per sequence per step. At batch 64 with
three speculative positions plus the bonus token, four rows per request is
4 MiB of masks moving toward the GPU each step: small against the step's
total traffic, but not free, and produced by CPU-side parser state updates
that must keep up with the engine's step rate. Compilation has its own cost
curve: compiling a large JSON Schema can take longer than the first request's
patience, which is why engines compile at admission time, cache automata
keyed by schema, and reuse them across requests that share a contract.

Tool and reasoning parsers add streaming semantics. A tool-call argument may be
incomplete for many tokens before it becomes valid JSON. The API must not emit a
final event too early, and cancellation must clean up parser state.

The streaming discipline is easiest to see character by character. While the
model emits the *name* of an argument, no consumer can act; partway through
its string *value*, every prefix is still syntactically open — a quote has
not closed, an escape may continue. A well-built streamer therefore emits
progress events keyed to parser states (argument started, value complete,
tool call complete), not to raw token arrivals, so a client never renders or
executes a half-argument. The same state that drives the mask drives the
events: one automaton, two consumers. Cancellation inherits the discipline —
a request cancelled mid-argument must release both its grammar automaton and
any buffered partial event, or the next request reusing pooled state starts
from someone else's open quote.

Constrained generation and speculation interact. A draft token forbidden by
the current grammar cannot be accepted. Masking may change proposal quality,
and verification must use the same constraint state as ordinary decode.

### Guided reading: one bitmask per speculative position

vLLM's `StructuredOutputManager` in
[`vllm/v1/structured_output/__init__.py`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/structured_output)
shows where the two halves of this chapter physically meet. The backend is a
choice — the package carries separate modules for xgrammar, outlines,
lm-format-enforcer, and a guidance backend — but the batching contract is
shared, and it is speculation-aware: the manager allocates its bitmask tensor
as `max_batch_size * (1 + max_num_spec_tokens)` rows, with the comment that
this is "one for each speculative position, and one more for the bonus
token." Every draft token needs its own legality check against grammar state
that does not exist yet, so the masks for a request's proposal are "stored
inline in the tensor and unpacked by the gpu runner."

Filling those masks is host work with its own budget. The manager fills them
on a thread pool in chunks of sixteen grammars, but only engages the pool
when more than 128 structured-output requests are scheduled and no
speculative tokens are in flight; otherwise it fills serially — the same
CPU-contention awareness the n-gram proposer showed, from the other side.
Grammars that finish are not dropped: `_fill_bitmasks` resets finished rows
to a full mask so a recycled batch index cannot inherit a stale constraint.
The reasoning-parser hooks (`should_fill_bitmask`,
`trim_reasoning_for_advance`) handle the streaming case — while a model is
inside its reasoning
span, the grammar for the final answer must wait. Chapter 5 counted the
mask's round trip into the step's latency budget; here is the machinery that
decides what is in it.

The pinned repositories expose these concerns in vLLM's
[`structured_output`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/structured_output)
and SGLang's
[`constrained`](https://github.com/sgl-project/sglang/tree/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/constrained)
packages.

## Worked example: acceptance is not speedup

Ordinary target decode costs 8 ms per token. A speculative step spends 3 ms on
drafting and 9 ms on verification. At 3.2 accepted tokens per step, it costs
3.75 ms per accepted token. At 1.3 accepted tokens, it costs 9.23 ms and loses
before accounting for extra memory.

Walk the arithmetic once more to find the hinge. The speculative round always
pays 3 + 9 = 12 ms and advances by however many tokens survive verification,
so its per-token cost is 12 divided by accepted tokens. Setting that equal to
the 8 ms baseline gives the break-even: 12 / a = 8 means a = 1.5 accepted
tokens per round. Above 1.5, speculation wins; below, it loses — and the
distance matters as much as the side. At 3.2 accepted the win is 2.1×, worth
real money; at 1.6 it wins by six percent, which a small regression in draft
latency or one unlucky traffic shift erases. A deployment that tracks
only "acceptance rate" sees 75 percent in both cases and cannot tell them
apart; accepted tokens per round — the metric this chapter has argued for —
is the number that crosses the line.

That produces an online decision rule: estimate accepted tokens and compare
draft plus verification plus capacity cost with ordinary target work. The
capacity cost has a walkable floor: a 1B-parameter draft model held in BF16
costs 2 GB, which against Chapter 4's 35 GiB admission budget is about three
fewer 8,000-token sequences per rank — 57 becomes 54, a five-percent
concurrency give before the first request arrives. A speculation win must
clear that bar too, not just the per-step arithmetic. Disable
speculation for short remaining outputs, low recent acceptance, memory
pressure, or graph-incompatible shapes. Acceptance rate remains an input to the
decision, not the result.

## Practice: find and explain the losing regime

Benchmark ordinary decode and two proposal strategies on predictable,
unpredictable, short-output, and high-concurrency traffic. Record draft and
verification time, accepted tokens, extra memory, graph dispatch, TTFT, ITL,
and output-distribution checks.

Use the numbers above to derive the break-even acceptance level, then add one
capacity penalty observed at concurrency. Write a rule for turning speculation
off. The worked calculation is in
[Appendix G](../appendices/g-worked-solutions.md#11-speculation-break-even).

We have now followed a request through one engine, from allocation to kernels
and decoding. Part III expands the same ideas across multiple accelerators and
machines.
