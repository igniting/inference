# 8. Kernels and Attention Backends

The scheduler has chosen 23 requests for the next step. Their sequences have
different lengths, their KV blocks are scattered through memory, and some use
an attention pattern that others do not. The model runner must turn this
irregular description into fast GPU work.

That work is performed by kernels: programs that execute across many GPU
threads. A model server may launch hundreds of kernels in one step, including
matrix multiplications, normalization, positional encoding, attention,
activation functions, expert routing, sampling, and memory copies.

## Why fewer operations can mean more speed

Framework code often expresses a calculation as several tensor operations.
Each operation may write an intermediate tensor to high-bandwidth memory, only
for the next operation to read it back.

Kernel fusion keeps intermediate values in registers or on-chip memory and
performs several logical operations in one launch. A fused normalization and
residual update, for example, can avoid round trips through device memory.

Fusion helps when memory traffic or launch overhead is the bottleneck. It can
hurt when the combined kernel uses too many registers, lowers occupancy, or
prevents a specialized library routine from running. “Fused” is not a synonym
for “faster.” It is a claim about a different movement and launch pattern.

## Attention is an I/O problem

The straightforward attention calculation creates a matrix of scores between
query and key positions, applies a softmax, and multiplies by values. Materializing
the full score matrix moves a great deal of data through GPU memory.

[FlashAttention](https://arxiv.org/abs/2205.14135) reorganizes the calculation
into tiles so that intermediate score regions remain in faster on-chip memory.
It computes exact attention while reducing reads and writes to high-bandwidth
memory. The important idea is broader than one kernel: algorithm design should
count data movement, not only arithmetic operations.

Serving attention is more complicated than the dense training case. Sequences
are ragged. Decode reads a growing history for one new query position. KV state
may be paged. Models use different masks, head layouts, latent representations,
sliding windows, or sparse patterns. A backend must understand both the model's
attention semantics and the engine's memory layout.

## Backends are compatibility decisions

An engine may integrate several attention implementations. Selection can depend
on device architecture, dtype, head dimension, page size, prefill or decode,
mask type, graph compatibility, and parallel plan.

If a preferred backend does not support one condition, the engine can reject
the configuration or fall back to another path. Silent fallback is dangerous
when the operator expects a particular performance profile. Startup logs and
metrics should identify the backend actually selected for each layer type.

At the pinned vLLM revision, the attention registry and implementations live
under
[`vllm/v1/attention/backends`](https://github.com/vllm-project/vllm/tree/5cecfc01375052698823fc401e31518fb32a981e/vllm/v1/attention/backends).
SGLang centralizes setup in
[`attention_backend_setup.py`](https://github.com/sgl-project/sglang/blob/e161bd1265a0082478b7f1c09f224a52d315dc71/python/sglang/srt/model_executor/model_runner_components/attention_backend_setup.py)
and maintains device- and model-specific backends elsewhere in the runtime.
The number of choices in both trees is evidence that one attention kernel does
not fit every serving shape.

## Matrix multiplication has shapes, not just FLOPs

Most model compute reduces to matrix multiplication, but two multiplications
with equal arithmetic counts can run at different speeds. Dimensions determine
whether tensor-core tiles are fully used. Alignment, dtype, transposition,
batching, and grouped execution all matter.

MoE layers make this visible. Each expert receives a different number of
tokens, so the engine often uses grouped GEMM to launch many expert
multiplications efficiently. A popular expert has a large matrix; another may
receive only a few rows. Padding can improve regularity while doing extra work.

Kernel libraries therefore offer families of implementations. Autotuning
measures candidate tiles or algorithms for representative shapes. The result is
usually cached because tuning itself is expensive. A production image should
decide whether tuning occurs during build, warm-up, or first traffic.

## Sampling can become expensive

Sampling appears small beside a transformer, but it touches a vocabulary that
may contain more than 100,000 entries for every active sequence. Applying
penalties, constraints, softmax, top-k or top-p selection, and random sampling
through separate kernels creates launches and memory traffic.

Fused sampling paths can help, especially for small models or large batches.
Structured-output masks add another tensor operation. The end-to-end effect
depends on how much of the step the model itself consumes.

## Test the kernel at three levels

A microbenchmark is useful for validating one operation. It controls shapes and
removes unrelated work. It does not show whether the engine can present those
shapes, whether conversion is needed, or whether the scheduler changes batch
composition.

For any proposed kernel change, measure three levels:

1. the isolated operation with representative shapes;
2. a complete engine step containing input preparation and surrounding work;
3. an end-to-end workload with queueing and output processing.

Suppose an attention kernel is 20 percent faster in isolation but requires a
page size that reduces useful prefix matches. The engine-step benchmark may
still improve while a multi-turn workload regresses. All three measurements
are necessary to explain the outcome.

Correctness tests should include awkward shapes: one-token decode, long
prefill, partially filled pages, uneven head dimensions, empty experts,
extreme logits, and masks with no valid continuation. Compare against a trusted
implementation with tolerances suited to the dtype. Performance cannot excuse
a model-semantic difference.

## Worked example: Amdahl meets the page size

Suppose a new attention kernel is 22 percent faster in isolation. Attention is
2.0 ms of a 5.0 ms engine step, so the maximum step saving is 0.44 ms. If the
new layout conversion costs 0.3 ms, the actual saving is 0.14 ms, or 2.8
percent—not 22 percent.

Now suppose the kernel requires 64-token pages instead of 16-token pages. More
tail waste and coarser prefix boundaries reduce cache capacity. Preemption or
recomputation can erase the remaining step win. The three levels answer
different questions: whether the operation improved, whether the step improved,
and whether users received more qualifying work.

## Practice: decide whether to enable the kernel

Evaluate the candidate above at isolated operation, complete step, and
production-trace levels. Include batch 1 and 32, contexts 127 and 4,096,
partially filled pages, and a multi-turn trace with prefix reuse. Measure
conversion, metadata, cache occupancy, preemption, output equivalence, and
goodput.

Write a conditional enablement rule rather than declaring a universal winner.
See [Appendix G](../appendices/g-worked-solutions.md#8-kernel-evaluation) for the
worked arithmetic.

Kernels reduce the cost of individual operations. The next chapter looks at a
different source of overhead: launching and specializing the whole operation
sequence.
