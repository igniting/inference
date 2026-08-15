# 4. Hardware Is a Topology

A machine specification says that a server has eight GPUs. That sounds
precise, but it leaves out the information an inference engineer needs most.

Can every GPU communicate directly with every other GPU? Do four devices sit
behind one CPU socket and four behind another? Is there one network interface
or several? Which links are shared? Eight identical accelerators can form very
different systems.

Hardware is not a bag of processors. It is a map of places where bytes can live
and paths along which bytes can move.

## Four limits to keep separate

Hardware discussions often collapse everything into “speed.” In practice, four
resources matter.

**Compute rate** describes how many arithmetic operations a device can perform
per second. **Capacity** describes how many bytes fit in a memory tier.
**Bandwidth** describes how quickly bytes move once a transfer is underway.
**Latency** describes how long a dependency takes, including the cost of
starting it.

A device can have enormous compute rate and still wait on memory. A network can
have high peak bandwidth but perform poorly for the tiny, frequent messages of
decode. A model can fit in device memory and still run out of space when the
engine reserves KV blocks, graph workspaces, and collective buffers.

Keep the four limits separate until measurements show which one governs the
workload.

## A practical roofline

Arithmetic intensity measures how much computation an operation performs for
each byte it moves from a chosen memory tier:

```text
arithmetic intensity = operations / bytes transferred
```

If a device can perform `C` operations per second and the relevant memory path
can move `B` bytes per second, a simple upper bound is:

```text
attainable rate <= min(C, B * arithmetic intensity)
```

Below the crossover point `C / B`, the operation is limited by moving data.
Above it, compute becomes the tighter ceiling. This roofline model is simple,
but it asks the right first question: should you reduce arithmetic, reduce
traffic, or improve overlap?

Apply the model at the boundary that matters. A decode layer may be limited by
GPU memory bandwidth while the entire model step waits on a cross-node
collective. A cache load may be fast from local storage and slow across PCIe.
There is not one roofline for the whole service.

## Why batching changes hardware efficiency

During a long prefill, large matrix operations reuse model weights across many
token positions. That gives the GPU substantial work for each byte of weights
it reads.

During decode, a small batch may read nearly the same weights to process only a
few new positions. The operation is more likely to be memory-bound. Adding
sequences to the batch allows one weight read to support more useful work.

This is why larger batches often improve throughput. It is also why throughput
and latency conflict: requests may need to wait until enough compatible work is
available, and a larger step itself takes longer. The scheduler chooses where
the service operates on this curve.

## Where the bytes live

An inference deployment may use a hierarchy that begins with registers and
on-chip scratch memory, continues through device caches and high-bandwidth
memory, and extends to host memory, local storage, remote storage, and durable
object storage.

Closer tiers are scarce and fast. Farther tiers provide more capacity at higher
access cost. Different objects deserve different treatment. Model weights are
large and repeatedly read. KV blocks grow with active sequences and may be
reused. Compiled graphs are expensive to recreate but tied to an execution
environment. Adapters and media embeddings have their own popularity patterns.

When checking whether a model fits, include the whole working set:

```text
weights
+ persistent request state
+ temporary activations
+ communication buffers
+ graph and compiler memory pools
+ allocator headroom
```

Parameter size alone is not a capacity plan.

## Links decide which parallel plans make sense

Within a host, devices may communicate over PCIe or a higher-bandwidth GPU
fabric. Across hosts, data may travel over RDMA-capable networks. Exact product
names change, but the design questions stay stable.

Which pairs have direct peer access? Which transfers cross a CPU root complex?
Which ranks share a switch or network rail? Can a transfer overlap the kernels
that surround it? What happens when every rank communicates at once?

Topology inventory is evidence, not a performance result. NVIDIA's
[DCGM topology guide](https://docs.nvidia.com/datacenter/dcgm/latest/learn/core-services/topology-and-links.html)
explicitly separates known CPU, PCIe, and NVLink relationships from active path
tests and observed traffic. Use the analogous inventory and diagnostic tools
for the platform being measured.

Different forms of model parallelism create different traffic. Tensor
parallelism commonly performs reductions or gathers at layer frequency. Expert
parallelism dispatches token representations to expert owners and combines the
results. Pipeline parallelism sends activations between neighboring stages.
Disaggregated prefill and decode move larger regions of persistent KV state.

The important quantities are message size, frequency, synchronization, and
path—not only total bytes.

## CPUs remain on the critical path

The GPU runs the model, but the CPU may parse requests, tokenize text,
preprocess media, make schedules, prepare metadata, coordinate transfers, and
turn outputs into stream events. Once GPU steps become short, Python work or a
host synchronization can take a large fraction of each iteration.

CPU placement also matters. A process can access memory attached to another
NUMA node or control a device behind another CPU socket. Tokenizer pools can
compete with network progress threads. Unified addressing can make memory
accessible without making it local or fast.

Measure CPU time, run-queue delay, memory placement, and synchronization beside
GPU utilization. A GPU that appears underused may be waiting for the host.

## Draw the physical map

Suppose you have 16 GPUs arranged as two groups of eight with fast links inside
each group and a slower network between them. A tensor-parallel group of eight
should usually fit within one fast island. Alternating its ranks across both
islands changes no arithmetic, but forces frequent collectives onto the slower
path.

Your topology drawing should follow each rank all the way out:

```text
rank -> accelerator -> local fabric -> CPU socket -> NIC -> switch -> rack
```

Now add traffic. Mark how many bytes cross each logical edge and how often. A
topology diagram without traffic is an inventory; adding traffic turns it into
a performance hypothesis.

Topology also defines failure. If one request needs every rank in a parallel
group, losing one rank can stop the group. Two replicas placed behind the same
power or network boundary do not provide the independence their count suggests.
A remote cache can improve warm-start performance and become a shared failure
dependency at the same time.

## Worked example: place before measuring

Place the dense model from Chapter 3 on two eight-GPU nodes with 80 GiB per GPU.
Fast links connect devices inside each node; the inter-node path is slower. A
four-way tensor-parallel replica holds about 35 GB of weights per rank before
overhead, and each rank holds roughly one quarter of the KV state.

Keep every four-rank group inside one fast-link island. Striping alternating
ranks across nodes changes no model arithmetic but moves layer-frequency
collectives onto the slower network. That is a topology error visible before a
profiler runs.

Predict long prefill to stress compute and attention traffic, batch-1 decode to
stress device-memory bandwidth and collective latency, and a 2.44-GiB KV move
to follow the slowest staging or network edge. The profiler's job is to falsify
those claims. Large CPU gaps during decode, for example, would reveal a host or
launch bottleneck the prediction omitted.

## Practice: write a falsifiable hardware prediction

Draw both nodes through GPU links, CPU sockets, NICs, and the connecting switch.
Place two four-way replicas, calculate per-rank weight and 8,000-token KV bytes,
and mark every collective and state-transfer path.

Predict the limiting resource for long prefill, batch-1 decode, and remote KV
load. For each prediction, name a counter or timeline observation that would
prove it wrong. Compare with
[Appendix G](../appendices/g-worked-solutions.md#4-topology-prediction).

That habit—predict, measure, explain—will be more useful than memorizing any
hardware table. Part II now follows a request through the software that turns
this topology into an executing service.
