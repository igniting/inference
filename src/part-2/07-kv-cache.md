# 7. Memory Management and Local Model State

Suppose a chat request has a maximum context length of 64,000 tokens. Reserving
a contiguous KV-cache region for all 64,000 positions would make growth easy,
but most requests would finish with much of the reservation unused. Reserving
only the current length saves memory, but the region must grow without moving
state that the GPU still needs.

The solution used by modern engines resembles virtual memory. Requests see a
logical sequence of positions. The engine backs those positions with fixed-size
physical blocks that do not need to be contiguous.

The resemblance is more than an analogy. Virtual memory solved the same three
problems: how to let consumers believe in a tidy private range while physical
storage is fragmented, how to share storage safely between consumers, and how
to reclaim it without asking anyone's permission at the wrong moment. Every
mechanism in this chapter — the block table, copy-on-write, reference counts,
eviction under live readers — has a direct ancestor in operating-system memory
management, which is useful because fifty years of OS practice tells you where
the bodies are buried.

## Logical tokens, physical blocks

Consider a block that holds 16 token positions. A 35-token sequence needs three
blocks. The first two are full and the last uses only three positions. When the
sequence grows, the allocator can attach another free block anywhere in the
cache.

**A block table separates logical sequence order from physical placement.**

```blockdiag
flowchart LR
    L0["Logical block 0"] --> P3["Physical block 3"]
    L1["Logical block 1"] --> P8["Physical block 8"]
    L2["Logical block 2"] --> P1["Physical block 1"]
    P3 --> K["Paged attention kernel"]
    P8 --> K
    P1 --> K
```


```text
logical positions:  [0 ........ 15][16 ....... 31][32 .. 34]
physical blocks:          7             2            19
```

A block table records this mapping. The attention kernel uses the table to find
keys and values. Because a request no longer needs one large contiguous region,
external fragmentation falls and the cache can support a changing set of
sequence lengths.

The [PagedAttention paper](https://arxiv.org/abs/2309.06180) describes this
virtual-memory-inspired design and the original vLLM implementation.

### Fragmentation, before and after paging

The win is easiest to see by pricing the alternative. Give every request a
contiguous reservation sized for the 64,000-token maximum. At 320 KiB of state
per token, each reservation commits about 19.5 GiB regardless of what the
conversation actually needs — a healthy 8,000-token exchange uses 2.44 GiB of
it, so roughly eight times the reservation sits idle even before short
requests are counted. On the deployment Chapter 4 walked, each device of a
four-way shard owes its quarter of every reservation, about 4.9 GiB against a
KV budget near 35 GiB: seven such reservations consume the pool, whether or
not their conversations ever grow. The same budget served fifty-six resident
conversations when allocation followed actual length — a factor-of-eight
difference in concurrency, purely reservation policy.

Paging attacks both fragmentation modes at once. Internal waste shrinks to the
tail of one block — at most fifteen unused positions out of sixteen, under
five megabytes per sequence and more than three orders of magnitude below the
contiguous case. External fragmentation stops being a scheduling concern: free memory
fragmented into scattered single blocks is perfectly usable, because the
allocator attaches them one at a time. The residual cost is metadata — larger
block tables crossing the scheduler-to-worker boundary every step — which is
why block size, the next section's subject, is a genuine trade and not a knob
to minimize blindly.

## Choosing a block size

Block size looks like a low-level allocator setting, but it influences the whole
engine.

Small blocks waste little space at the end of a sequence and allow fine-grained
prefix matches. They create larger block tables and more allocation work. Large
blocks reduce metadata and may suit an attention kernel better, but waste more
tail space and only reuse prefixes at coarser boundaries.

Transfers add another consideration. Moving many tiny blocks can pay protocol
overhead repeatedly. A large transfer unit may move unused positions.

Backend constraints sometimes narrow the choice. SGLang's current
[attention-backend guide](https://docs.sglang.io/docs/advanced_features/attention_backend)
documents page-size requirements for several implementations and explains the
trade-off between kernel performance and prefix-match granularity. The correct
size is part of an execution plan, not a universal constant.

The interactions run wider than allocation. Chapter 6's prefill chunking
interacts with block size because a chunk that stops mid-block leaves a
partially filled block whose publication timing the lifecycle rules govern;
Chapter 15's cross-node transfers price block size directly, since the
transfer unit determines how much protocol overhead repeats per hop. When a
team changes block size, it is quietly renegotiating with the scheduler, the
kernel, and the transfer layer simultaneously — which is why the setting
belongs to the execution plan review, not to a config default.

One worked contrast shows how sharply the choice bites. At sixteen-token
blocks, a 100-token conversation fills seven blocks and wastes at most fifteen
positions — under five megabytes of tail. At 256-token blocks, the same
conversation fills one block and abandons 156 positions: about forty-nine
megabytes of state held for thirty-two megabytes of use, waste exceeding the
payload. For a chat fleet dominated by short requests, the large block is
difficult to justify at any batch size; for a deployment of long documents
where every sequence fills dozens of blocks, the tail is rounding error and
the smaller tables and kernel-friendly pages win. The block size that
maximizes reuse granularity is a function of the length distribution the
service actually serves — one more quantity Chapter 2's workload records
exist to pin down.

## A block has a lifecycle

Allocation is not merely “free” or “used.” A block can be allocated while a
step is about to write it, valid and owned by a running request, valid but kept
only for reuse, or waiting for an asynchronous transfer to finish.

**Reusable blocks move through ownership states before returning to the pool.**

```blockdiag
flowchart LR
    F["Free"] -->|allocate| W["Private and writable"]
    W -->|GPU complete| S["Sealed"]
    S -->|publish| R["Reusable and referenced"]
    R -->|evict index| D["Draining"]
    D -->|reference count zero| F
```

The first diagram is the indirection that makes everything else possible:
logical order is a fiction the kernel resolves through the table, so growth,
sharing, and release never require moving bytes. The second diagram is the
discipline that keeps the fiction honest — a block becomes visible to others
only after its writer is provably finished, and leaves the pool only after its
last reader is provably gone. The table below names the four moments where
that discipline is most often violated, and what each violation looks like
from outside.

| Cache concern | Identity or invariant | Observable signal |
| --- | --- | --- |
| legal reuse | tokens, positions, weights, adapter, format | matched tokens by namespace |
| branching | sealed blocks shared; tail copied | copy-on-write count |
| cancellation | in-flight blocks remain pinned | deferred-release age |
| eviction | visibility removed before storage | references after index removal |


```text
free -> reserved -> being written -> valid and owned
     -> valid and reusable -> evictable -> free
```

Asynchronous engines must be conservative. If step `t` can still write a block,
the allocator cannot hand that address to step `t+1` for another request. A
remote sender cannot release a block while a transfer still reads it. Engines
may defer release until the relevant stream or acknowledgement proves the old
use has finished.

Reference counts protect ownership. Cache policy decides how long an unowned,
valid block remains available for reuse. Combining those two ideas risks
evicting state that a live request still needs.

The distinction matters most at transitions. A block moving from private to
reusable crosses from “one owner's truth” to “many readers' assumption,” and
the engine must prove the writer finished — GPU event, stream sync, or
acknowledged transfer — before flipping the bit in between. Skipping the proof
works in benchmarks, where steps complete predictably, and fails in production
exactly when a cancellation, preemption, or stalled transfer makes completion
order surprising. This is why the lifecycle diagram's middle states exist at
all: they are the proof obligations, made explicit.

Cancellation exercises every state at once, which is why it makes the best
test case. The cancelled request's in-flight blocks must stay pinned — not
published, not freed — until the completion event proves the GPU finished
writing them; only then may they seal into reusable state or drop back to the
pool. Engines that release eagerly show a characteristic signature: rare
garbage tokens in unrelated requests, appearing only under cancellation load,
because a reallocated address was still being written by a ghost. The
“deferred-release age” signal in the table above is the operational probe for
this — how long blocks wait between their request ending and their last use
provably finishing — and an age that grows with load is an engine telling you
its proofs are falling behind its execution.

Two eviction refinements round out the picture. First, eviction policy
operates at block granularity here, but token-level policies also exist:
score each cached token by its estimated future importance (attention-magnitude
schemes are the canonical example) and drop low-scoring positions while
keeping the block. They trade correctness structure for capacity — a dropped
token changes attention outputs for everything after it, so unlike block
eviction under prefix identity, the result is no longer equivalent to a
cache miss. Fine for lossy compression deployments that accept it; wrong for
anything that promised Chapter 22-style output equivalence. Second, eviction
under adapters must consider the adapter dimension too: a block reused under
a different adapter is not a hit at all, which is the subject of the next
section.

### Serving many adapters at once

Adapter serving is where cache identity becomes a scheduling problem. Take a
low-rank adapter at rank 16 on Atlas's hidden size of 8,192: each layer
carries two matrices of `8192 × 16` BF16 values, `2 × 8192 × 16 × 2 = 512 KiB`
per layer, about 40 MiB across 80 layers — four orders of magnitude smaller
than the 140 GB base model. That ratio is the whole economics of adapter-dense
serving: a fleet can hold thousands of adapters resident for the cost of one
extra base replica, and Chapter 17's 800 ms cold-adapter load is not I/O wait
but the price of not having the weights paged where the batch needs them.

The serving designs follow from the arithmetic. Because an adapter's working
set is tiny and its compute is two thin matrix products per layer, engines
keep every adapter resident and batch across *different* adapters in one
step: the base weights are read once regardless, each sequence adds its own
low-rank products, and the per-request extra arithmetic is a few percent of
the step. The hard parts are the ones this chapter already built for KV
state: the activation buffers the low-rank paths need must be paged and
sized per batch composition, the block table must carry which adapter each
sequence runs under so a mixed step never mixes identities, and CUDA-graph
capture (Chapter 9) must either fix the adapter set per graph or read
pointers dynamically — a captured graph with baked adapter weights silently
serves the wrong model, the same failure class as Chapter 20's stale-weight
caches. When an interviewer asks how one replica can serve a thousand
customer-specific models, the answer is this section: adapters make weights
a per-request cache problem, and everything from Chapter 7 applies with
40 MiB objects instead of gigabyte ones.

## Reusing a prefix

The support assistant in Chapter 1 begins every conversation with the same
system prompt. Once the model has processed that prompt, later requests can
reuse its KV state instead of repeating the prefill.

But matching text is not enough. The cached state depends on the model and
weight version, tokenizer, exact token IDs, positions, adapter, attention
configuration, and any multimodal features. A service may also include a tenant
namespace or cache salt to prevent sharing across isolation boundaries.

Engines commonly index prefixes in one of two ways. A chain of hashes identifies
successively longer blocks. A radix tree stores shared token paths and makes
branches explicit. Hash indexing works naturally with page-granular and
distributed lookup. Radix indexing makes structured sharing easy to see. Both
need collision handling and version separation.

The [SGLang paper](https://arxiv.org/abs/2312.07104) introduced RadixAttention
for reuse across structured language-model programs. In the pinned code,
SGLang's
[`radix_cache.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/mem_cache/radix_cache.py)
contains prefix matching, insertion, request caching, and eviction. vLLM's
[`kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/core/kv_cache_manager.py)
coordinates request allocation and cached-block lookup.

Read the two implementations and the identity rules from earlier in this
section stop being abstract. In SGLang's `RadixCache`, the lookup key is a
`RadixKey` carrying the token IDs plus an optional `extra_key`, and the
`match_prefix` docstring states the namespace policy outright: entries with
identical leading tokens but different `extra_key` values are “kept disjoint
and never share prefix nodes,” which is how LoRA adapters, sampling salts, and
cache versions partition the tree without changing the token content. Matching
is page-aligned — keys are truncated to a multiple of `page_size` before
lookup, so reuse boundaries obey the allocator's granularity — and when a
match ends inside a stored segment, the method splits that node once to expose
a precise boundary, a structural refinement that duplicates no data. The same
walk refreshes access timestamps, which is how lookup feeds the configured
eviction strategy: using a prefix is itself a cache-policy event. One more
detail shows how deep identity goes: the key may be converted to a bigram
view when speculative decoding is active, because draft-token patterns change
what a reusable segment means — even the matching representation bends to the
execution mode.

vLLM's `get_computed_blocks` enforces the complementary rule on the consumer
side: returned cached blocks must be full, and `max_cache_hit_length` is set to
`request.num_tokens - 1`. The comment explains why — if every prompt token hit
the cache, there would be nothing left to run through the model, and no logits
would exist — so the last token is always recomputed. Because allocation is
block-aligned, that single recomputed token can drag a whole block with it; the
comment flags this honestly as a known inefficiency. The same function shows
multi-cache realities surfacing in the interface: the coordinator finds the
longest hit per state group, and when sparse-retention groups such as Mamba or
sliding-window layers lag behind the full-attention groups, the result carries
a `shared_prefix_boundary` marking the junction all groups can agree on.

Allocation closes the loop in `allocate_slots`, whose docstring draws the
block layout of a request as `<comp> | <new_comp> | <ext_comp> | <new> |
<lookahead>` — already-computed, newly-hit, externally-delivered, to-be-run,
and speculative-reserved regions laid end to end. Two of its parameters are
admission policy hiding inside an allocator: `full_sequence_must_fit` forces
the whole sequence to fit now, closing the loophole where chunked prefill
checks only whether the first chunk fits and strands the rest; and
`reserved_blocks` keeps free blocks aside for in-flight sequences, so an
asynchronous KV load cannot consume pages a prefilling request is relying on.
The boundary between “memory manager” and “scheduler” runs straight through
this signature — which is why Chapter 6 ended by promising this chapter.

## Sharing and copy-on-write

Parallel samples or beam candidates can share the blocks that represent their
common prompt. When they generate different tokens, their new state diverges.
The request shape that triggers this is ordinary: one API call asking for four
completions becomes, at Chapter 5's boundary, four execution requests whose
prefixes are identical by construction — sharing is not an optimization the
caller requests but a consequence of what the engine notices.

Full, immutable blocks can remain shared. If two sequences share a partially
filled block and one needs to write into it, that sequence receives a private
copy. This is copy-on-write. It saves memory, but cancellation and completion
must update references exactly once.

The same branching appears in multi-turn conversations. A shared system prompt
may be extremely valuable; thousands of rare branches may not be. The cache
needs an eviction policy, not merely the ability to retain everything.

### What branching actually costs

The savings are easiest to trust with numbers. Take four parallel samples of
one prompt — 1,000 tokens of shared context. Without sharing, four private
sequences hold `4 × 1,000 × 320 KiB ≈ 1.22 GiB`; with full blocks shared,
the pool holds one copy of `313 MiB`, and the three siblings are pure
saved capacity. Divergence starts costing only when children write: the first
token each child emits lands in the shared partial tail, which must be
copied — sixteen positions, about five megabytes per child, once. From there
each child pays only for what makes it different: after fifty divergent
tokens, a child owns roughly four extra blocks plus the copied tail, tens of
megabytes against the original hundreds. The economics that make beam search
and best-of-n sampling affordable are exactly these: share everything
immutable, pay only at divergence, and let reference counts settle who still
needs each block.

Copy-on-write also interacts with the identity rules in a way worth noticing:
a branch that copies its tail inherits the parent's provenance up to the fork
point and owns everything after. If the branch later changes adapter mid-life
— rare, but tools do this — the inherited portion stays valid only under the
original namespace, so the engine must either forbid the change or re-key the
branch's identity from the fork point onward. Systems that skip this check
produce caches that serve correct-looking state assembled from two
incompatible worlds.

## A hit rate can be misleading

Least-recently-used eviction is a reasonable starting point. It does not know
how expensive a prefix is to recompute, how likely it is to return, how many
bytes it occupies, or whether another tier already holds a copy. Even its
central quantity needs interpretation in a tree: when a shared system prompt
sits at the root of thousands of branches, every hit below it touches the
root, and naive recency bookkeeping makes the root permanently immortal while
the leaves — where workload change actually shows first — evict first. The
clock is part of the policy, not an implementation detail beneath it.

Imagine two cached prefixes. One contains 10,000 tokens and is reused once an
hour. The other contains 100 tokens and is reused every second. A raw request
hit rate favors the small prefix. Saved prefill computation may favor the large
one. Saved work per byte may produce a third answer.

Useful cache metrics include matched tokens, compute time avoided, bytes held,
bytes transferred, eviction churn, and the effect on request latency. Hit rate
alone is not enough.

### Pricing two prefixes

Give the pair numbers. The 10,000-token system prompt saves about `0.035 ms ×
10,000 = 350 ms` of prefill work each time it hits; at once an hour, that is
350 milliseconds of GPU time avoided per hour, bought with `10,000 × 320 KiB
≈ 3.05 GiB` of residency. The 100-token preamble saves 3.5 ms per hit, but at
one hit per second it avoids about 12.6 seconds of prefill per hour while
occupying barely 31 MiB. Measured per byte, the small prefix is thousands of
times more productive — yet a deployment with spare memory should absolutely
keep the big one, because 3.05 GiB sitting idle in an uncongested pool costs
nothing and buys 350 ms off every conversation start.

The ranking flips exactly when memory becomes scarce. Under pressure, those
gigabytes have an opportunity cost — Chapter 4's admission walk priced a
rank-share of long sequences at roughly 0.61 GiB apiece — and evicting the
hourly giant to admit another resident conversation may raise goodput even
though the giant's individual hits feel valuable. That is the real lesson of
the misleading hit rate: cache value is a function of current scarcity, not a
property of the entry, and any policy fixed at insertion time will eventually
be answering yesterday's question.

A cost-aware policy falls out of the same framing without much ceremony.
Score each cached prefix by its expected savings rate — reuse probability ×
tokens matched × per-token prefill cost, divided by bytes held — refresh the
probability from observed hits, and evict lowest score first when the pool
needs blocks. The formula's inputs are all things Chapter 2 said to record:
matched tokens, hit frequency, bytes. Two refinements matter in practice:
scores must decay, because a workload shift makes yesterday's hot prefix
today's dead weight; and admission needs the same test as eviction, or the
pool fills with newly inserted prefixes that a scoring pass would immediately
evict — churn that costs metadata work on every step for zero reuse.

## Modern models have more than one cache shape

Some models mix full attention with sliding-window attention. Others include
recurrent or state-space layers. They may share KV state between layers or use
compressed latent attention.

Each layer type can require a different amount of state and a different
retention rule. The longest prefix available for one group of layers may not be
valid for another. A correct engine needs a cache specification for each state
group and must choose a prefix that all required groups can support.

At the pinned vLLM revision, these ideas appear in
[`kv_cache_interface.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/kv_cache_interface.py)
and the cache coordinator. SGLang has separate memory-pool and radix-cache paths
for sliding-window, Mamba, and unified layouts. The names will change; the
underlying requirement comes from the model.

The coordination cost is subtle: a mixed-model hit is the *minimum* across
groups, so one lagging layer type silently truncates reuse for everyone. A
model whose sliding-window group retains four thousand positions caps every
hit there, even when the full-attention groups hold thirty-two thousand
reusable positions — the window group's oldest state no longer exists, and
no amount of caching in the other groups restores it. An
engine that reports prefix hits only from the largest group will flatter
itself while requests quietly recompute sliding-window state the report
claimed was cached. Honest multi-group metrics count the junction — the
boundary all groups reached — not the best single group.

## Worked example: publication before reuse

A request produces a final partial block and is cancelled while the GPU write
is still in flight. Making that block immediately visible creates two hazards:
a reader can observe incomplete data, and cleanup can reallocate an address the
GPU still uses.

Keep the block private and pinned until the completion event. Then either seal
and publish it under the cache policy or discard it. A branch shares sealed
full blocks but copies a partial tail before writing. Eviction removes lookup
visibility first and frees storage only after references reach zero.

The two hazards name the two proof obligations precisely. A reader observing
incomplete data is a torn-state failure — the block became visible before its
writer finished — and the completion event is the proof that prevents it. An
address being reallocated under a live writer is a use-after-free failure —
the block returned to the pool before its last device use ended — and the
pinned-until-event rule prevents that one. Every lifecycle state in this
chapter exists to discharge one of these two obligations, and any shortcut
that skips a proof is betting that the race it opens never wins.

Content identity also needs a boundary. If token 511 changes, at most the first
511 tokens match. A different adapter or model version invalidates the produced
state even when token IDs are identical. Tenant policy may forbid otherwise
valid cross-tenant reuse.

## Quick KV budget worksheet

Use this worksheet to calculate your deployment's KV memory budget.
Replace the Atlas numbers with your model's actual constants.

```text
Step 1: Weight memory per rank
  weights = 140 GB (Atlas BF16) / TP_degree
  Example: 140 / 4 = 35 GB per rank

Step 2: Non-KV overhead per rank
  activations ≈ 0.5–1.5 GB (depends on batch size and model)
  graph pool  ≈ 0.5–2.0 GB (depends on captured buckets)
  framework   ≈ 0.5–1.0 GB
  Subtotal:   ≈ 1.5–4.5 GB

Step 3: Available KV memory per rank
  available = GPU_memory - weights - overhead
  Example: 80 - 35 - 3.0 = 42 GB per rank

Step 4: Maximum tokens in KV cache
  KV per token per rank = KV_bytes_per_token / TP_degree
  Example: 320 KiB / 4 = 80 KiB per rank per token
  Max tokens = available / (KV per token per rank)
  Example: 42 GB / 80 KiB ≈ 550,000 tokens

Step 5: Maximum concurrent sequences
  max_sequences = max_tokens / average_context_length
  Example at 4K context: 550,000 / 4,000 = 137 sequences
  Example at 32K context: 550,000 / 32,000 = 17 sequences

Step 6: Set max-num-seqs to 85% of Step 5
  Example at 4K: 137 × 0.85 ≈ 116
  Example at 32K: 17 × 0.85 ≈ 14
```

The 85% margin prevents preemption storms. If your actual traffic has
variable context lengths, use your p90 context length in Step 5. The
decision checklist in Appendix D walks this calculation with additional
considerations for adapters, speculative decoding, and quantized KV.

## Practice: construct a cache safety matrix

Starting from one 512-token prefix, vary exactly one of token content, adapter,
image feature, position scheme, model version, tenant, and physical block
layout. State the legal reusable prefix and why.

Then test cancellation during a write, branching from a partial block, and
eviction with a live reader. Assert unpublished-state isolation, copy-on-write,
eventual reference release, and output equivalence with caching disabled. The
worked matrix is in [Appendix G](../appendices/g-worked-solutions.md#7-kv-cache-correctness-matrix).
