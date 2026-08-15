# 19. Inference for Reinforcement Learning

In online reinforcement learning for language models, inference does not serve
an end user. It generates experience for a trainer.

The policy model produces one or more responses for each prompt. A reward or
verifier scores them. The trainer updates the policy, and the new weights return
to the inference workers for another round. The loop can repeat thousands of
times.

```text
prompts -> rollout generation -> reward -> training update
   ^                                      |
   +----------- new policy weights -------+
```

This workload changes the engine's lifecycle. Model weights are no longer
immutable for the duration of the service.

## Rollouts arrive in groups and waves

Training algorithms often request several completions for the same prompt.
Their shared prefix creates a strong cache opportunity. Output lengths can be
highly variable, especially for reasoning tasks. A group may not be ready for
training until enough valid samples finish.

The longest rollout can hold up a synchronous batch while most workers become
idle. Partial-rollout protocols allow the orchestrator to cancel, pause, or
accept a subset according to the algorithm. The inference engine must report
which policy version and sampling configuration produced every trajectory.

Scheduler fairness also changes. Advancing a nearly complete prompt group may
unblock the trainer sooner than serving equal tokens across all groups. The
right policy depends on the training algorithm's data dependencies.

## Colocate or disaggregate?

Training and inference can share GPUs in alternating phases. Colocation avoids
dedicated idle pools and can transfer weights locally. It also requires careful
memory handoff because optimizer state, training activations, inference weights,
and KV cache may not fit together.

A disaggregated design gives training and rollout separate pools. Both can work
concurrently, but weights must cross the network and rollout data can become
stale.

The [AReaL paper](https://arxiv.org/abs/2505.24298) studies a fully asynchronous
system where rollout workers continue generating while training workers update
the model. Its result is not permission to ignore staleness; the system and
algorithm explicitly manage it.

## Sleeping is memory coordination

An inference engine can temporarily release or offload weights and KV state so
a colocated trainer can use the device. Waking restores the required resources
without rebuilding the whole process and distributed environment.

Different sleep levels may retain weights in host memory, discard KV state, or
release both. The order matters during an update. Freeing KV memory before
receiving new weights can reduce peak usage. The engine should not resume
scheduling until weights and cache allocations are ready.

vLLM's official [sleep-mode documentation](https://docs.vllm.ai/en/stable/features/sleep_mode/)
describes releasing model and KV memory and selectively waking resources for
RL workflows. SGLang exposes comparable sleep, wake, and weight-update controls
through its engine and scheduler paths.

## Weight update is a versioned transaction

A safe update has a beginning, a data phase, and a commit point.

1. Stop admitting model steps that would overlap the change.
2. Establish the transfer group and expected parameter metadata.
3. Move or refit every shard.
4. Verify completion across ranks.
5. invalidate state derived from old weights;
6. publish the new policy version and resume.

If one rank receives only part of the update, the model is not a slightly stale
version—it is a corrupt mixture. The protocol must fail closed and recover all
ranks to one version.

The new weights invalidate KV and encoder state produced by the old policy.
They may also invalidate compiled artifacts if shapes or modules changed.
Weight-only updates with an identical architecture can retain some graph
structures, but the service should prove rather than assume compatibility.

Current vLLM documentation describes a four-phase pluggable
[weight-transfer protocol](https://docs.vllm.ai/en/stable/training/weight_transfer/).
At the pinned code snapshot, update and lifecycle routes appear under the
development RLHF and sleep APIs. SGLang's model-runner components include
weight updater and remote transport paths, while scheduler components coordinate
lifecycle state.

## On-policy does not mean one global pause

Strictly synchronous rollout keeps every sample tied to one policy version, but
creates bubbles between generation and training. Fully asynchronous rollout
keeps hardware busy and trains on older policies.

Several middle grounds exist. Complete prompt groups can move to training while
later groups finish under the same version. The system can allow a bounded
number of active policy versions. It can prioritize frontier groups that will
complete the next training batch.

The algorithm determines which staleness is legal. The serving system must make
version and group boundaries observable enough to enforce it.

## Numerical agreement matters more in training

The trainer may recompute token log probabilities and compare them with values
reported during rollout. Different kernels, precisions, templates, or batch
shapes can create mismatches. Importance ratios can amplify small differences.

Record token IDs, masks, positions, model version, sampling state, and log
probabilities with clear semantics. Decide whether the trainer uses inference
engine values or recomputes them. Test long responses, padding boundaries,
structured outputs, and MoE routing.

Batch-invariant or deterministic modes help debugging, but may cost throughput.
Use them to isolate differences even if the final production configuration is
less strict.

## Worked example: prepare, then commit

Rollouts use policy version 41 while the trainer produces version 42 in inactive
buffers. Every inference rank validates tensor shapes and checksums, then
reports `prepared(42)`. Only after all ranks prepare does the coordinator commit
generation 42. Ranks swap buffers, invalidate version-dependent caches and
graphs, run a health forward pass, and reopen admission.

If one rank fails mid-copy, no commit is published. Prepared ranks keep version
41 active and retry or discard their inactive buffers. This is a distributed
transaction because a tensor-parallel group with mixed weights is not a valid
model.

## Practice: fail one update safely

Trace rollout admission, reward, training, pause, staged transfer, prepare,
commit, invalidation, health check, and wake. Attach the policy version to every
trajectory, cache entry, graph, and message.

Fail rank 3 during transfer and show why no mixed group resumes. Then delay the
trainer and define queue-byte and policy-lag admission bounds. Compare with
[Appendix G](../appendices/g-worked-solutions.md#19-policy-update-transaction).

Inference inside training is still a serving system, but the customer is an
algorithm with stronger version and reproducibility requirements. Chapter 20
examines another long-lived customer: the interactive session.
