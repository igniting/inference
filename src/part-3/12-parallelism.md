# 12. Parallelism as Data Movement

A model no longer fits on one GPU. The obvious response is to add another GPU.
The difficult question is what to split.

You can split weights, layers, tokens, experts, attention heads, sequence
positions, or complete requests. Each choice reduces one device's work or
memory by moving something between devices. Parallelism is therefore best
understood as a data-movement plan.

## Start with replication

If the model fits on one device, the simplest scale-out design is replication.
Each replica holds the complete model and serves independent requests. This is
data parallelism in its inference form.

Replication adds capacity without putting a collective on the critical path of
one request. It also duplicates weights and fragments warm state across
replicas. A router must decide where requests go. For latency-sensitive models
that fit, replication is the baseline against which more complicated plans
should be justified.

## Tensor parallelism splits a layer

Tensor parallelism divides the matrices inside a layer across ranks. A common
transformer plan shards one linear operation by output dimension and another by
input dimension. Partial results are combined with an all-reduce or
reduce-scatter.

The benefit is immediate: each rank stores and multiplies only a shard. The cost
is also immediate: ranks communicate at layer frequency. Fast links and large
matrix shapes can make the exchange worthwhile. Small decode batches or slow
cross-node links can make synchronization dominate.

The foundational [Megatron-LM paper](https://arxiv.org/abs/1909.08053) explains
intra-layer tensor parallelism for transformer models. Its training context is
different, but the partition and collective reasoning carries into inference.

Tensor parallelism can lower single-request latency when the communication is
cheaper than the removed computation. It should not be a reflexive default.

## Pipeline parallelism splits layers

Pipeline parallelism assigns consecutive layer ranges to different stages. An
activation moves from one stage to the next. Each stage stores only its layers,
so pipeline parallelism solves model capacity without a collective inside every
layer.

The pipeline must remain occupied. If only one microbatch is present, later
stages wait while the first stage begins and earlier stages wait after their
work moves on. These empty periods are pipeline bubbles.

Serving makes scheduling difficult because sequences enter and leave
dynamically. Chunked prefills can provide pipeline work, while short decode
steps may expose bubbles. Pipeline parallelism is attractive across links where
one activation transfer per stage is cheaper than repeated tensor collectives,
but it needs enough concurrent work.

## Splitting sequence positions

Long contexts can make attention state or computation too large for one rank.
Context or sequence parallelism divides token positions across devices. Each
rank computes a portion of attention and the partial results are combined.

Ring-style attention passes key/value regions around ranks. Ulysses-style
methods exchange tensor dimensions so each rank can compute local attention.
Decode-context parallelism can stripe the stored context across ranks while a
small number of new queries attend to all shards.

These methods trade memory capacity and attention compute for communication.
They become attractive when context state, not weights, is the limiting
resource.

## Expert and attention parallelism

MoE models allow experts to be distributed independently from the attention
layers. Expert parallelism places different experts on different ranks and
moves token representations to their selected owners.

Some deployments replicate or data-parallelize attention while expert layers
span a larger group. This is often called attention data parallelism. It avoids
tensor collectives in attention and uses the expert all-to-all as the main
cross-replica exchange.

The model no longer has one parallel size. It has a mesh of dimensions.

## Compose a rank mesh

Suppose a deployment has 64 GPUs and chooses:

```text
data parallel = 4
tensor parallel = 2
expert parallel = 8
```

The product is 64 only if those axes are independent in the implementation.
Some systems define expert parallel within or across data-parallel groups;
others couple sizes. Pipeline and context axes add further constraints.

Write down what each coordinate means. A rank might be identified as:

```text
(replica 2, tensor shard 1, expert shard 6)
```

Then list the communication groups. Which ranks participate in layer
all-reduces? Which participate in expert dispatch? Which share a KV partition?
This exercise catches configurations that multiply cleanly but communicate
poorly.

Both source snapshots centralize group construction and rank state: vLLM in
[`parallel_state.py`](https://github.com/vllm-project/vllm/blob/5cecfc01375052698823fc401e31518fb32a981e/vllm/distributed/parallel_state.py)
and SGLang in
[`parallel_state.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/distributed/parallel_state.py).
Reading these files is often the fastest way to learn what a framework's
parallel-size arguments actually compose.

## Map frequent traffic to fast links

A logical mesh becomes a serving topology only when it is mapped to hardware.
Keep frequent, latency-sensitive collectives inside the fastest fabric when
possible. Put communication that uses larger, less frequent transfers on
slower links.

For every parallel dimension, create a small ledger:

| Dimension | What is split? | What moves? | How often? | Synchronization? |
| --- | --- | --- | --- | --- |
| Tensor | weights and partial activations | reductions or gathers | per layer | strong |
| Pipeline | layer ranges | activations | per stage | pipeline dependency |
| Context | sequence positions | KV or partial attention | per layer or step | strong |
| Expert | experts | routed token activations | per MoE layer | all-to-all and stragglers |
| Data | requests | usually no model tensors | per request/control update | weak |

Attach message sizes from the target model and batch. The labels alone cannot
predict performance.

## A decision procedure

Begin by asking whether the weights and required state fit on one device. If
they do, test replication first. If weights do not fit, choose between splitting
layers and splitting tensors based on links, latency, and available concurrency.
If context state does not fit, add a sequence or decode-context dimension. If
experts dominate model size, distribute experts and model the all-to-all.

Next, check the workload. Low-concurrency interactive traffic is sensitive to
collective latency and pipeline bubbles. High-throughput offline work can fill
larger parallel groups. Long-context traffic changes the fraction of memory and
communication due to state.

Finally, measure the plan at several batch shapes. Parallel efficiency is not a
fixed property of the model.

For an exercise, design two legal plans for the same model and cluster. Draw
their rank meshes and physical mappings. Calculate the messages on each link
for one prefill and one decode step. Predict which plan wins in each phase
before running it.

The next chapter focuses on the parallel dimension with the most irregular
traffic: experts.
