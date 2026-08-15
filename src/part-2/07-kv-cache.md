# 7. Memory Management and the KV Cache

Suppose a chat request has a maximum context length of 64,000 tokens. Reserving
a contiguous KV-cache region for all 64,000 positions would make growth easy,
but most requests would finish with much of the reservation unused. Reserving
only the current length saves memory, but the region must grow without moving
state that the GPU still needs.

The solution used by modern engines resembles virtual memory. Requests see a
logical sequence of positions. The engine backs those positions with fixed-size
physical blocks that do not need to be contiguous.

## Logical tokens, physical blocks

Consider a block that holds 16 token positions. A 35-token sequence needs three
blocks. The first two are full and the last uses only three positions. When the
sequence grows, the allocator can attach another free block anywhere in the
cache.

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

## A block has a lifecycle

Allocation is not merely “free” or “used.” A block can be allocated while a
step is about to write it, valid and owned by a running request, valid but kept
only for reuse, or waiting for an asynchronous transfer to finish.

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

## Sharing and copy-on-write

Parallel samples or beam candidates can share the blocks that represent their
common prompt. When they generate different tokens, their new state diverges.

Full, immutable blocks can remain shared. If two sequences share a partially
filled block and one needs to write into it, that sequence receives a private
copy. This is copy-on-write. It saves memory, but cancellation and completion
must update references exactly once.

The same branching appears in multi-turn conversations. A shared system prompt
may be extremely valuable; thousands of rare branches may not be. The cache
needs an eviction policy, not merely the ability to retain everything.

## A hit rate can be misleading

Least-recently-used eviction is a reasonable starting point. It does not know
how expensive a prefix is to recompute, how likely it is to return, how many
bytes it occupies, or whether another tier already holds a copy.

Imagine two cached prefixes. One contains 10,000 tokens and is reused once an
hour. The other contains 100 tokens and is reused every second. A raw request
hit rate favors the small prefix. Saved prefill computation may favor the large
one. Saved work per byte may produce a third answer.

Useful cache metrics include matched tokens, compute time avoided, bytes held,
bytes transferred, eviction churn, and the effect on request latency. Hit rate
alone is not enough.

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

## Worked example: publication before reuse

A request produces a final partial block and is cancelled while the GPU write
is still in flight. Making that block immediately visible creates two hazards:
a reader can observe incomplete data, and cleanup can reallocate an address the
GPU still uses.

Keep the block private and pinned until the completion event. Then either seal
and publish it under the cache policy or discard it. A branch shares sealed
full blocks but copies a partial tail before writing. Eviction removes lookup
visibility first and frees storage only after references reach zero.

Content identity also needs a boundary. If token 511 changes, at most the first
511 tokens match. A different adapter or model version invalidates the produced
state even when token IDs are identical. Tenant policy may forbid otherwise
valid cross-tenant reuse.

## Practice: construct a cache safety matrix

Starting from one 512-token prefix, vary exactly one of token content, adapter,
image feature, position scheme, model version, tenant, and physical block
layout. State the legal reusable prefix and why.

Then test cancellation during a write, branching from a partial block, and
eviction with a live reader. Assert unpublished-state isolation, copy-on-write,
eventual reference release, and output equivalence with caching disabled. The
worked matrix is in [Appendix G](../appendices/g-worked-solutions.md#7-kv-cache-correctness-matrix).

With memory organized, the engine can present irregular batches to the GPU.
Chapter 8 looks at the kernels that turn those mappings into useful work.
