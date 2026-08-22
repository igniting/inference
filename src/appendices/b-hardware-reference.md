# Appendix B. Hardware and Communication Reference

This appendix is a checklist for investigating a deployment. It avoids product
performance tables because hardware and software support change faster than the
principles in the main text.

## Memory tiers

| Tier | Typical role | Questions to ask |
| --- | --- | --- |
| Registers/on-chip memory | kernel tiles and intermediates | Does fusion increase register pressure? Is the tile shape efficient? |
| Device cache | recently accessed data | Is access regular enough to benefit? |
| Accelerator HBM | weights, KV state, activations, graph pools | What is usable capacity after reserves? What is sustained bandwidth? |
| Host DRAM | offload, preprocessing, staging | Which NUMA node owns it? Is it pinned? What crosses PCIe? |
| Local storage | cold weights, KV backup, artifacts | What are random and sequential behavior? Is capacity shared? |
| Remote memory/storage | distributed cache and model source | What are network, consistency, and failure costs? |

Accessible memory is not necessarily local memory. Unified addressing and
coherence simplify programming while physical movement still affects latency
and bandwidth.

## Topology discovery checklist

Record:

- device model, memory capacity, power mode, and supported numerical formats;
- peer-to-peer connectivity and link width between every device pair;
- CPU sockets, cores, NUMA nodes, and memory attachment;
- NICs, rails, link rate, RDMA support, and device affinity;
- switch and rack boundaries, oversubscription, and failure domains;
- local storage devices and paths used for models or cache;
- driver, runtime, communication-library, firmware, and kernel versions.

On NVIDIA systems, tools such as `nvidia-smi topo -m` and NCCL topology logs can
help reveal device relationships. Use the corresponding vendor tools on other
platforms. Verify with a bandwidth and latency test; a discovered link does not
prove the expected path is active.

## Collective operations

| Collective | Result | Common inference use |
| --- | --- | --- |
| Broadcast | one rank's data reaches all ranks | configuration or weight distribution |
| All-reduce | reduction result reaches all ranks | tensor-parallel partial outputs |
| Reduce-scatter | reduced result is sharded | tensor/sequence-parallel output shards |
| All-gather | shards are assembled on all ranks | tensor or sequence reconstruction |
| All-to-all | each rank sends a distinct piece to every rank | expert dispatch and combine |
| Point-to-point | one source communicates with one destination | pipeline stages and cache transfer |

Message size and synchronization determine behavior. Measure small decode
messages and large prefill messages separately. Aggregate bus bandwidth does
not reveal a slow rank or overloaded rail.

## Communication questions by parallel method

**Tensor parallelism:** How many collectives occur per layer? Are they inside a
fast local fabric? Can communication overlap adjacent computation?

**Pipeline parallelism:** What activation crosses each boundary? How many
microbatches are needed to fill the pipeline? Which stage is slowest?

**Expert parallelism:** What is the dispatch distribution by rank? Are prefill
and decode using appropriate transport modes? Which expert creates a straggler?

**Context parallelism:** Is KV state circulated, gathered, or reduced? How does
communication grow with context? How are partial softmax statistics combined?

**Disaggregation:** How many state bytes move per request? Can block layout or
parallel-size differences require gathering and scattering? What pins the
source and destination during transfer?

## Memory-budget worksheet

For each rank, fill in:

| Item | Steady bytes | Peak bytes | Lifetime | Reclaim policy |
| --- | ---: | ---: | --- | --- |
| Weight shard | | | deployment | unload or sleep |
| KV/request state | | | request/session | finish, preempt, offload |
| Encoder state | | | request/reuse window | evict or transfer |
| Activations | | | model step | immediate |
| Graph pools | | | process/configuration | restart or recapture |
| Collective buffers | | | operation/process | backend managed |
| Compiler/autotune workspace | | | warm-up/operation | backend managed |
| Safety reserve | | | continuous | not allocated |

Run the worksheet at the largest legal shape and during warm-up. Peak phases
can occur in a different order from steady serving.

## Order-of-magnitude classes

Interviews and design reviews move faster when orders of magnitude are
already in your head. These are *classes*, not product claims — each
generation moves the boundaries — but the ratios between rows are the durable
part. Treat them as declared planning figures in the book's sense.

| Quantity | Class | Notes |
| --- | --- | --- |
| Accelerator HBM bandwidth | 2–4 TB/s | the number that sets decode's roofline |
| Dense BF16/FP16 arithmetic peak | ~1–2 PFLOPS per accelerator | tensor-core peak; MFU divides against this |
| Intra-island link bandwidth | 400–900 GB/s per direction | NVLink-class fabrics |
| Cross-host network | tens to hundreds of GB/s (RDMA) | an order below intra-island |
| PCIe host path | tens of GB/s | why host staging is a copy, not a transfer |
| Kernel launch overhead | 3–10 µs | why graphs exist (Chapter 9) |
| Host–device sync | ~10 µs | why per-step syncs are budgeted, not free |
| Inter-region network RTT | tens of milliseconds | why Chapter 16 routes before crossing regions |

Two habits make the table useful rather than trivia. First, keep the
*ratios*: intra-island bandwidth is roughly a hundred times the host path,
and arithmetic peak is roughly a thousand times HBM bytes per second — those
two ratios explain most of this book's architecture. Second, re-derive
crossovers rather than memorizing them: peak divided by bandwidth gives the
arithmetic-intensity knee (Chapter 4), and it moves every generation even
when the ratio structure does not.

## Common traps

- Measuring device-to-device bandwidth without the application's concurrent
  compute and message sizes.
- Mapping logical ranks in a way that sends frequent collectives across nodes.
- Ignoring CPU affinity and memory placement.
- Reserving all free HBM for KV state before graph capture.
- Assuming a quantized format has a native kernel on every device.
- Treating link bandwidth as available to every pair simultaneously.
- Placing redundant replicas in the same network or power failure domain.
- Using an average transfer size that hides many small control operations.
