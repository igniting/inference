# 22b. Debugging Inference in Practice

Chapter 22 teaches you to design experiments that produce trustworthy numbers.
Chapter 23 teaches you to build the observability that makes incidents
diagnosable. This chapter sits between them: it walks three real debugging
sessions end-to-end, each starting from a symptom an operator would see in
production and ending at a verified fix. The goal is not to catalogue every
possible failure but to demonstrate the *method* -- split the symptom into
candidate causes, use evidence to eliminate branches, and verify the fix
against the same measurement that raised the alarm.

Every walkthrough uses the Atlas constants: 140 GB of BF16 weights across a
TP4 deployment (35 GB per rank), 320 KiB of KV state per token, 0.035 ms per
prefill token, 45 ms decode step, 600 ms TTFT target, 150 ms ITL ceiling.
Commands and metrics reference vLLM and SGLang at their pinned SHAs; the
investigation method transfers to any engine.

## Visual map

**A debugging session is a directed search, not a tour of dashboards.**

```blockdiag
flowchart LR
    S["Symptom"] --> H["Hypotheses ranked by likelihood"]
    H --> E["Evidence: metric, trace, or profile"]
    E --> D{"Hypothesis confirmed?"}
    D -->|Yes| F["Fix and verify"]
    D -->|No| H
    F --> V["Regression gate"]
```

**Each walkthrough follows the same five-step discipline.**

```blockdiag
flowchart TB
    subgraph Method
        direction TB
        S1["1. Observe the symptom precisely"]
        S2["2. List candidate causes"]
        S3["3. Split with targeted evidence"]
        S4["4. Identify the root cause"]
        S5["5. Fix, verify, add the signal"]
    end
    S1 --> S2 --> S3 --> S4 --> S5
```

| Walkthrough | Symptom | Misleading signal | Actual cause |
| --- | --- | --- | --- |
| High TTFT | p95 TTFT jumps from 480 ms to 1,400 ms | GPU utilization drops, suggesting underload | KV cache pressure triggers preemption and re-prefill |
| OOM restarts | Workers restart with OOM, no request pattern | Memory looks stable between crashes | CUDA graph pool grows for unseen shapes |
| ITL spikes | p99 ITL hits 300+ ms, median stays 48 ms | Decode kernel looks slow | Long prefill chunks in mixed batches extend step time |

## Walkthrough 1: high TTFT under load

### The symptom

Monday morning traffic ramp. Atlas's p95 TTFT rises from its baseline of
480 ms to 1,400 ms -- more than double the 600 ms target. The on-call
engineer opens the GPU dashboard and sees utilization has *dropped* from 72
to 41 percent. The instinct is to add replicas. Resist it: falling
utilization means accelerators are starving for work, not drowning in it.
Adding capacity treats a symptom and hides the cause.

### Step 1: router, queue, or engine?

The first split separates three regions of the request path. Each has a
distinct signal.

```blockdiag
flowchart LR
    subgraph Region
        direction LR
        R["Router / load balancer"] --> Q["Engine admission queue"]
        Q --> P["Prefill execution"]
    end
    R --> R1["Ingress latency histogram"]
    Q --> Q1["Queue age and depth"]
    P --> P1["Prefill duration histogram"]
```

Check the router first. If the load balancer is slow to assign requests,
TTFT rises without any engine involvement. Pull the ingress-to-engine
latency from the trace span:

```bash
# Query the last 15 minutes of router-to-engine latency from Prometheus
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(router_forward_duration_seconds_bucket[5m]))'
```

If this value is under 20 ms, the router is not the bottleneck. Move on.

### Step 2: locate the queue with vLLM metrics

vLLM exposes Prometheus metrics at its `/metrics` endpoint. Two are
immediately diagnostic:

```bash
# How many requests are sitting in the waiting queue right now?
curl -s http://localhost:8000/metrics | grep 'vllm:num_requests_waiting'

# What does the TTFT distribution look like?
curl -s http://localhost:8000/metrics | grep 'vllm:time_to_first_token_seconds'
```

In this incident, `vllm:num_requests_waiting` reads 47 -- far above the
normal operating point of 3 to 8. The oldest waiting request has been
queued for 1.1 seconds. That queue age directly consumes TTFT budget:
a request that waits 1.1 seconds before prefill begins cannot meet a
600 ms TTFT target regardless of how fast prefill runs. The queue is the
proximate cause. The question becomes: *why is the queue backing up?*

Two candidate mechanisms explain queue growth: prefill is taking longer
than expected (requests drain slowly), or KV cache pressure is preventing
admission (requests cannot enter the batch even though the GPU has time).

### Step 3: profile the prefill path

Check whether prefill itself has slowed. Pull the engine's prefill
duration:

```bash
curl -s http://localhost:8000/metrics | grep 'vllm:e2e_request_latency_seconds'
```

Compare against the expected cost. A 2,000-token prompt on Atlas costs
roughly `2000 * 0.035 = 70 ms` of prefill compute. If chunked prefill is
enabled with a 512-token chunk budget, that prompt crosses four engine
steps of approximately `512 * 0.035 = 18 ms` of prefill work each,
interleaved with decode. The total wall-clock prefill time will be longer
than 70 ms because each step also carries decode work, but the individual
chunk durations should stay near 18 ms.

Three conditions inflate prefill time:

1. **Long prompts without chunking.** A 16,000-token prompt processed in
   one shot takes `16000 * 0.035 = 560 ms` and holds the GPU for that
   entire duration, stalling every in-flight decode request. Check whether
   chunked prefill is enabled and whether the chunk budget is sized
   appropriately.

2. **Missing CUDA graph capture.** If the batch shape has not been
   captured as a graph, the engine falls back to eager execution. Check
   for graph-fallback log entries or the `vllm:num_graph_captures` counter
   not incrementing.

3. **Interference from mixed steps.** Under chunked prefill, decode tokens
   share each engine step with a prefill chunk. If the chunk budget is too
   large relative to the decode population, step time stretches and ITL
   suffers -- but that would appear as an ITL problem, not a TTFT problem.
   In this case, the queue growth points elsewhere.

In this incident, prefill durations look normal. The queue is growing not
because requests drain slowly, but because they are not being admitted.

### Step 4: check KV cache pressure

This is the critical split. Pull the memory metrics:

```bash
curl -s http://localhost:8000/metrics | grep 'vllm:gpu_cache_usage_perc'
curl -s http://localhost:8000/metrics | grep 'vllm:num_preemptions_total'
```

KV cache usage reads 94 percent. Preemption count is climbing: 23
preemptions in the last five minutes, versus a baseline of zero. The
mechanism is now clear.

Atlas's KV budget per rank is approximately 35 GB (the accelerator memory
minus the 35 GB weight shard, minus activation and graph overhead). At
320 KiB per token, that supports roughly `35 * 1024 * 1024 / 320 = 114,688`
tokens of live KV state per rank. If the current batch has 40 active
sequences averaging 2,800 tokens each, that is `40 * 2800 = 112,000`
tokens -- 97 percent of capacity.

When a new request arrives and no blocks are free, the scheduler must
preempt: it evicts the KV state of one or more lower-priority requests,
freeing their blocks, and admits the new request. The evicted requests
re-enter the queue and must re-prefill from scratch when they are later
re-admitted. This re-prefill inflates their TTFT enormously -- a request
that was 80 percent complete in decode loses all its KV state and starts
over. Worse, the re-prefill consumes GPU time, slowing other admissions,
creating a cascade.

The preemption storm explains both symptoms: TTFT rises because requests
bounce between the queue and partial execution, and GPU utilization falls
because the engine spends cycles re-computing KV state it already
produced.

### Step 5: the fix and verification

The immediate remediation is to tighten admission. Reduce the maximum
concurrent sequences from 40 to 28, keeping KV occupancy below 80 percent
of capacity:

```bash
# Restart with a lower max-num-seqs to leave KV headroom
python -m vllm.entrypoints.openai.api_server \
    --model atlas-70b \
    --tensor-parallel-size 4 \
    --max-num-seqs 28 \
    --enable-chunked-prefill
```

After the change, verify:

1. `vllm:gpu_cache_usage_perc` stays below 0.80.
2. `vllm:num_preemptions_total` stops climbing.
3. `vllm:num_requests_waiting` returns to the 3-to-8 range.
4. p95 TTFT drops below 600 ms.

The longer-term fix addresses *why* sequences grew long enough to exhaust
the cache: check whether the output length distribution shifted (users
submitting longer conversations), whether prefix caching is releasing
blocks correctly, or whether the max-context-length setting is broader
than the workload requires.

## Walkthrough 2: memory pressure and OOM

### The symptom

Over a 48-hour period, two of Atlas's four TP4 workers restart with CUDA
out-of-memory errors. The crashes happen at different times of day with no
obvious correlation to request volume or type. Between crashes, GPU memory
metrics look stable. The instinct is to blame a memory leak in user
requests. That is almost never the cause.

### Step 1: establish the memory budget

Before hunting leaks, know what the budget *should* be. For one Atlas
rank:

| Component | Size | Notes |
| --- | --- | --- |
| Model weights | 35 GB | 140 GB / 4 ranks, BF16 |
| KV cache pool | ~35 GB | Sized to fill remaining memory |
| Activation buffers | ~1.5 GB | Peak intermediate tensors for max batch |
| CUDA graph pool | ~2-4 GB | Captured graphs for common shapes |
| Framework overhead | ~1-2 GB | PyTorch allocator, NCCL buffers, misc |
| **Total** | **~75-78 GB** | On an 80 GB device |

The margin between the budget and the device limit is 2 to 5 GB. Any
component that grows beyond its expected allocation will eventually
trigger OOM. The question is: which component is growing?

### Step 2: track allocated versus reserved

Start with the coarse signal:

```bash
# Snapshot GPU memory state across all ranks
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total \
    --format=csv,noheader,nounits
```

This shows total memory consumption but does not distinguish PyTorch
allocations from CUDA driver state. For finer granularity, enable
PyTorch's memory snapshot:

```python
import torch

# Enable memory history tracking (do this before model load)
torch.cuda.memory._record_memory_history(max_entries=100000)

# ... run workload ...

# Dump snapshot to file for analysis
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

Load the snapshot in PyTorch's memory visualizer
(`torch.cuda.memory._snapshot()`) to see allocation timelines. Two
patterns distinguish the common causes:

- **A leak** appears as a monotonic increase in allocated memory that
  never returns to baseline, even when the request queue is empty.
- **A pool growth** appears as step increases in reserved (but not
  necessarily allocated) memory that coincide with specific events.

In this incident, the snapshot shows reserved memory growing in discrete
2 GB jumps roughly every 8 to 12 hours.

### Step 3: check for KV block leaks

A KV block leak occurs when a request completes but its blocks are not
returned to the free pool. The block manager's reference count stays
nonzero, and the blocks remain allocated forever. Over hours, the free
pool shrinks.

Check the free block count over time:

```bash
# Track free blocks via the metrics endpoint
curl -s http://localhost:8000/metrics | grep 'vllm:gpu_cache_usage_perc'
```

If cache usage ratchets upward even during low-traffic periods (when
sequences complete faster than they arrive), blocks are leaking. In this
incident, cache usage returns to baseline during off-peak hours. The leak
is not in the KV pool.

### Step 4: check CUDA graph pool growth

CUDA graphs capture a fixed sequence of GPU operations for replay without
CPU launch overhead. Each captured graph allocates a private memory pool
for the tensors it uses during execution. The pool is sized to the
captured shape -- batch size, sequence length, and intermediate buffer
requirements.

The critical detail: if a batch shape arrives at runtime that was *not*
captured during warmup, the engine must either fall back to eager
execution or capture a new graph on the fly. On-the-fly capture allocates
a new graph pool. If the shape is rare, the pool sits mostly idle,
consuming memory without proportionate benefit.

Check for graph captures after startup:

```bash
# Look for graph capture events in the engine log
grep -i "capturing\|cuda graph\|graph capture" /var/log/vllm/engine.log
```

In this incident, the logs show new graph captures at irregular
intervals -- each coinciding with one of the discrete memory jumps in the
snapshot. The trigger is an unusual batch composition: when the scheduler
happens to assemble a batch of 17 decode tokens plus a 384-token prefill
chunk (a shape not seen during warmup's graph capture sweep), the engine
captures a new graph. Each capture allocates roughly 1.5 to 2.5 GB of
graph pool memory that persists for the process lifetime.

Over 48 hours, three or four such captures accumulate 6 to 10 GB of
graph pool memory beyond the startup budget, consuming the 2 to 5 GB
margin and eventually triggering OOM on the next allocation spike.

### Step 5: the fix and verification

The fix has two parts.

First, expand the warmup capture set to cover all reachable batch shapes.
The graph capture sweep should enumerate the combinations of decode
batch sizes and prefill chunk sizes that the scheduler can actually
produce, not only the "standard" batch sizes:

```python
# In the engine configuration, specify explicit capture shapes
# that cover the scheduler's actual output range
--enforce-eager  # Temporary: disable graphs to stop the bleeding
```

Then, after computing the full set of reachable shapes from the
scheduler's budget and chunk configuration, re-enable graph capture with
an explicit shape list or rely on the engine's padded capture buckets
(Chapter 9). If the engine supports a maximum graph pool size, set it:

```bash
# Limit graph memory to prevent unbounded growth
# (engine-specific; check documentation for the exact flag)
export VLLM_GRAPH_RESERVED_MEM=4GiB
```

Second, add a memory-growth alert. The signal is not total memory usage
(which fluctuates with batch load) but the gap between PyTorch's
`reserved_memory` and `allocated_memory`. A growing gap that does not
shrink during idle periods indicates pool fragmentation or graph
accumulation:

```bash
# Monitor the reserved-allocated gap
python -c "
import torch
for i in range(torch.cuda.device_count()):
    r = torch.cuda.memory_reserved(i) / 1e9
    a = torch.cuda.memory_allocated(i) / 1e9
    print(f'GPU {i}: reserved={r:.1f}GB allocated={a:.1f}GB gap={r-a:.1f}GB')
"
```

Verification: after restarting with the expanded capture set, confirm
that no new graph captures appear in the log after warmup completes, and
that the reserved-allocated gap remains stable over 72 hours.

## Walkthrough 3: tail ITL spikes

### The symptom

Atlas's median inter-token latency holds steady at 48 ms -- healthy,
given the 45 ms decode step plus sampling and streaming overhead. But
p99 ITL spikes to 300 ms or higher, well above the 150 ms ceiling.
The spikes are intermittent and do not correlate with request volume
in an obvious way. The instinct is to profile the decode kernel. That
is almost certainly not the problem -- a kernel that is fast at the
median does not become 6x slower at the tail without a discontinuity.

### Step 1: correlate spikes with batch composition

ITL measures the gap between consecutive tokens in a single response
stream. A spike means one particular engine step took much longer than
usual. The question is whether the spike belongs to the model (the
compute was slow) or to the batch (the step included extra work).

Under chunked prefill, each engine step may contain a mix of decode
tokens and a prefill chunk. The prefill chunk adds compute: a 512-token
chunk costs roughly `512 * 0.035 = 18 ms` of additional work in the
step. If the chunk budget is set to 1,024 tokens, the chunk alone adds
`1024 * 0.035 = 36 ms`, stretching the step from 45 ms to approximately
81 ms. Add scheduling overhead, attention over the growing context, and
the decode portion: a step near 90 ms is plausible, but 300 ms is not
explained by a single standard chunk.

Pull the per-step composition from the engine's iteration metrics:

```bash
# Check the token composition of recent engine steps
curl -s http://localhost:8000/metrics | grep 'vllm:num_tokens_prefill'
curl -s http://localhost:8000/metrics | grep 'vllm:num_tokens_decode'
```

In this incident, the spikes correlate with steps where the scheduler
admits a very long prefill chunk -- 4,096 tokens or more from a new
request with a long prompt, processed without chunking because the
request's priority forced immediate admission. That chunk costs
`4096 * 0.035 = 143 ms` of prefill compute, pushing the mixed step
past 200 ms. If two such admissions coincide, the step reaches 300+ ms.

Every decode-phase request in that step sees the entire step duration as
its ITL for that token. The spike is not in their computation; it is in
the time they waited for the step to complete.

### Step 2: profile individual engine steps

To confirm, capture a short trace with PyTorch profiler during a period
when spikes are occurring:

```python
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./traces'),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(20):
        engine.step()
        prof.step()
```

Load the trace in TensorBoard or Chrome's `chrome://tracing`. Look for
the step that corresponds to the ITL spike. In the CUDA stream, you
will see the attention kernels for the prefill chunk dominating the
step -- their combined duration matches the expected `chunk_tokens *
0.035 ms` cost, confirming that the compute is correct but the chunk
is too large.

### Step 3: check for GC pauses and CPU-side bottlenecks

Not all ITL spikes come from the GPU. Python's garbage collector can
pause the engine's main loop, and CPU-side output processing (detokenization,
streaming, sampling) can hold up the next step.

Check for GC pauses:

```python
import gc

# Enable GC debugging to see collection events
gc.set_debug(gc.DEBUG_STATS)
```

If GC collections correlate with ITL spikes (visible in logs as
"gc: collecting generation 2" events taking 10+ ms), the fix is to
disable automatic GC and collect manually between steps or during
idle periods:

```python
gc.disable()
# Collect explicitly during known idle points
```

In this incident, GC pauses account for a few spikes in the 160 to
180 ms range but not the 300+ ms outliers. The primary cause remains
the oversized prefill chunks.

### Step 4: check for collective stragglers in TP groups

In a TP4 deployment, every engine step ends with an all-reduce across
four ranks. The step completes only when the slowest rank finishes. A
rank that is slow -- due to thermal throttling, PCIe contention, memory
bandwidth saturation, or an unrelated process on the host -- extends the
step for all ranks.

Check for rank imbalance:

```bash
# Compare per-GPU utilization and clock speeds
nvidia-smi --query-gpu=index,clocks.current.sm,utilization.gpu,temperature.gpu \
    --format=csv,noheader
```

If one rank's SM clock is throttled below the others (for example,
1,200 MHz versus 1,410 MHz due to thermal limits), its compute takes
proportionally longer, and the all-reduce synchronization point extends
every step.

In this incident, all four ranks show similar clocks and utilization.
The straggler hypothesis is eliminated.

### Step 5: the fix and verification

The root cause is the scheduler admitting prefill chunks larger than the
chunk budget during priority overrides. The fix enforces chunking
unconditionally:

```bash
# Set a strict chunk budget that limits per-step prefill work
python -m vllm.entrypoints.openai.api_server \
    --model atlas-70b \
    --tensor-parallel-size 4 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512
```

With a 512-token chunk budget, the maximum prefill contribution per step
is `512 * 0.035 = 18 ms`. The mixed-step ceiling becomes approximately
`45 + 18 = 63 ms`, well below the 150 ms ITL target.

Verification:

1. p99 ITL drops below 150 ms and stabilizes near 65 to 70 ms.
2. Median ITL remains near 48 ms (the chunk does not meaningfully
   affect small steps).
3. TTFT may increase slightly because long prompts now take more steps
   to prefill. Check that p95 TTFT stays within the 600 ms budget;
   if not, add a replica rather than increasing the chunk budget.

The trade-off is explicit: smaller chunks protect ITL at the cost of
higher TTFT for long prompts. Chapter 6's budget arithmetic predicted
this dial; the debugging session confirmed its operating point.

## The profiling toolkit

Profiling is for *explaining* a result, not for measuring one. A
profile run perturbs the system it observes -- tracing adds overhead,
memory tracking consumes memory, and both change scheduling. Run
profiles on a staging replica with representative traffic, never on a
production worker under real load.

### nvidia-smi: the first look and its limits

```bash
# Continuous monitoring at 100 ms intervals
nvidia-smi dmon -s pucvmet -d 100
```

What nvidia-smi provides: GPU utilization percentage, memory usage,
temperature, clock speeds, power draw, PCIe throughput. These are
useful for coarse triage -- "is the GPU doing anything?" -- and for
detecting thermal throttling or memory exhaustion.

What nvidia-smi *cannot* tell you: whether GPU utilization is useful
work. The utilization counter reports the fraction of time at least one
kernel was running on the device. A kernel that performs redundant
recomputation, executes a fallback path, or pads a half-empty batch
all register as 100 percent utilization. A system at 95 percent
utilization and 30 percent goodput is broken; nvidia-smi will call it
healthy. Use utilization to detect absence of work, never to confirm
quality of work.

### torch.profiler: CPU and GPU timeline

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(10):
        engine.step()

# Export for Chrome trace viewer or TensorBoard
prof.export_chrome_trace("engine_trace.json")
```

The trace shows CPU and GPU activity on parallel timelines. Look for:

- **Gaps between GPU kernels.** These indicate CPU-side launch overhead,
  Python processing, or synchronization stalls.
- **Long CPU spans during model steps.** These suggest tokenization,
  sampling, or output processing is on the critical path.
- **Memory allocation events.** Unexpected allocations during steady-state
  inference indicate missing pre-allocation or graph fallbacks.

### NSight Systems: kernel-level analysis

For deeper investigation, NVIDIA NSight Systems captures kernel
launches, memory transfers, NCCL collectives, and PCIe activity:

```bash
nsys profile -t cuda,nvtx,osrt \
    --stats=true \
    --force-overwrite=true \
    -o engine_profile \
    python -m vllm.entrypoints.openai.api_server \
        --model atlas-70b \
        --tensor-parallel-size 4
```

NSight answers questions that torch.profiler cannot:

- Is the all-reduce overlapping with compute, or is it serialized?
- Are CUDA memory copies (H2D, D2D) appearing where they should not?
- Is kernel occupancy limited by register pressure or shared memory?

The cost is significant: NSight captures produce multi-gigabyte trace
files and the instrumentation overhead can alter step timing by 10 to
20 percent. Use it for targeted investigation after coarser tools have
narrowed the search.

### vLLM /metrics endpoint

vLLM exports Prometheus-format metrics at `/metrics`. The most
diagnostic metrics for debugging:

| Metric | What it tells you | Normal range (Atlas) |
| --- | --- | --- |
| `vllm:num_requests_running` | Active sequences in the batch | 8-32 |
| `vllm:num_requests_waiting` | Queue depth | 0-8 |
| `vllm:num_requests_swapped` | Sequences swapped to CPU | 0 |
| `vllm:gpu_cache_usage_perc` | KV cache occupancy | 0.3-0.8 |
| `vllm:num_preemptions_total` | Cumulative preemptions | 0 (should not grow) |
| `vllm:time_to_first_token_seconds` | TTFT histogram | p95 < 0.6 s |
| `vllm:inter_token_latency_seconds` | ITL histogram | p99 < 0.15 s |
| `vllm:num_generation_tokens_total` | Output throughput counter | Steady growth |

Pull the full set with `curl -s http://localhost:8000/metrics` and
pipe through `grep vllm:` to filter engine-specific counters from
Python runtime metrics.

### SGLang /get_server_info

SGLang exposes runtime state through a JSON endpoint:

```bash
curl -s http://localhost:30000/get_server_info | python -m json.tool
```

The response includes the current batch composition, memory usage,
cache hit rates, and scheduler state. For debugging, the most useful
fields are the active request count, the pending queue length, and the
per-request token counts -- these let you reconstruct what the
scheduler is doing without reading engine logs.

### When to profile

| Situation | Tool | Where to run |
| --- | --- | --- |
| Initial triage | nvidia-smi, /metrics | Production (read-only) |
| Queue and latency analysis | Prometheus queries, traces | Production metrics store |
| Step-level investigation | torch.profiler | Staging with replay traffic |
| Kernel and collective analysis | NSight Systems | Staging, isolated node |
| Memory leak investigation | torch memory snapshot | Staging or canary replica |

The boundary is clear: *read* production signals, *profile* staging
replicas. A torch.profiler capture on a production worker adds 15 to
30 percent overhead per step, which violates SLOs for every request
served during the capture. Use the production metrics to identify the
*regime* (batch size, queue depth, traffic pattern), reproduce that
regime on staging, and then profile the reproduction.

## Common pitfalls

The table below collects failure patterns that recur across inference
deployments. Each row names the symptom an operator sees, the wrong
diagnosis that intuition suggests, and the actual root cause that
evidence reveals.

| # | Symptom | Wrong first guess | Actual root cause | Confirming evidence |
| --- | --- | --- | --- | --- |
| 1 | p95 TTFT spikes during traffic peaks | GPU is too slow; add replicas | KV cache preemption forces re-prefill of evicted requests | `num_preemptions_total` climbing; cache usage > 90% |
| 2 | OOM crashes with no request pattern | Memory leak in model code | CUDA graph captured for warmup-unseen batch shapes; each capture allocates a persistent pool | Graph capture log entries after startup; reserved-allocated gap grows in steps |
| 3 | p99 ITL exceeds target, median is fine | Decode kernel regression | Oversized prefill chunks in mixed engine steps inflate step time for co-scheduled decode requests | Per-step token composition shows prefill chunks > budget |
| 4 | GPU utilization is 95% but throughput is low | Hardware is at capacity | Batch is padded or decode slots hold completed-but-unreleased sequences; compute is wasted on non-useful work | `num_requests_running` much higher than actual active requests; completed sequences with lingering state |
| 5 | Latency degrades after deploying new model version | New model is slower | CUDA graphs from previous version are invalidated; engine recompiles during serving | Graph capture events in logs; step times return to baseline after warmup completes |
| 6 | One replica is consistently slower than others | Bad GPU / hardware lottery | NUMA misalignment: model weights cross socket boundary, doubling memory access latency | `numactl --hardware` shows memory on remote node; `nvidia-smi topo -m` shows suboptimal placement |
| 7 | Prefix cache hit rate drops to zero after restart | Cache is broken | Cache index is ephemeral; after restart, all prefixes must be re-computed before matches resume | Hit rate recovers over minutes as traffic rebuilds the index |
| 8 | Requests time out but no errors in engine logs | Engine crashed silently | Deadlock in NCCL collective: one TP rank received different batch composition, collective hangs forever | Process is alive but stuck; `py-spy` shows all threads blocked in NCCL wait |
| 9 | Memory usage slowly climbs over hours | KV blocks leaking | Python reference cycles prevent garbage collection of request metadata; accumulated objects consume host memory | `gc.get_objects()` count grows monotonically; forcing `gc.collect()` recovers memory |
| 10 | TTFT is fine at low load but degrades linearly with concurrency | Prefill is compute-bound | Queue wait dominates: at high concurrency, new requests wait behind decode-heavy batches that leave no prefill budget | Queue age grows linearly with offered load; prefill compute time per request is constant |
| 11 | Throughput drops after enabling speculative decoding | Speculation has too much overhead | Draft model and target model share a memory pool; speculation reduces KV budget, lowering batch concurrency | KV cache usage rises; `max-num-seqs` effective limit drops; throughput falls from concurrency loss, not speculation cost |
| 12 | Streaming responses stall for 2-3 seconds mid-generation | Network issue between server and client | Engine preempted the request to admit a higher-priority one; KV state was evicted and must be recomputed before generation resumes | Preemption counter increments at stall time; the request's TTFT metric shows a second prefill phase |

## Debugging as a practice

Each walkthrough above followed the same discipline: observe the symptom
precisely, list candidate causes, split with evidence, identify the root
cause, and verify the fix against the original measurement. The method
is more valuable than any individual diagnosis because inference systems
surface new failure modes as workloads, models, and engines evolve.

Two habits make the method sustainable. First, after every resolved
incident, add the confirming signal to the monitoring stack. The
preemption counter in Walkthrough 1, the reserved-allocated gap in
Walkthrough 2, and the per-step token composition in Walkthrough 3 were
all available before the incident -- but nobody was watching them. Each
incident teaches you which signal to promote from "available" to
"alerted." Second, maintain a staging replica that can reproduce
production traffic patterns. The profiling toolkit is powerful but
invasive; without a safe place to use it, operators are forced to choose
between diagnosing the problem and serving traffic. A staging replica
with trace replay eliminates that choice.

Chapter 23 builds the operational framework -- alerting, runbooks,
deployments -- that turns these individual debugging skills into a
team practice.
