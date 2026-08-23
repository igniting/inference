# Final Editorial Review

Date: August 23, 2026

This review compares the finished manuscript with the supplied 259-page PDF of
Philip Kiely's *Inference Engineering*. The comparison concerns coverage,
teaching method, and reader experience. The reference was not treated as a
source of reusable prose, examples, figures, or organization.

## What the reference taught us

The reference is most effective when it begins with a recognizable engineering
problem, builds intuition in ordinary language, and introduces terminology only
after the reader has something concrete to attach it to. Its strongest pages
use short paragraphs, purposeful figures, and equations that answer a decision.

The first manuscript pass was less reader-friendly: it opened with abstraction,
compressed several mechanisms into survey-like paragraphs, repeated the same
prefill/decode and constrained-generation explanations, and placed visual maps
before the concepts they were meant to explain. Those were structural problems,
not cosmetic ones.

## Changes incorporated from the critique

- Chapter 0 now follows one request through tokenization, prefill, KV creation,
  decode, and streaming before the book introduces distributed abstractions.
- Adapter serving is a numbered Chapter 12 rather than an interstitial chapter.
- Parallelism, MoE, disaggregation, distributed caching, and routing form a
  continuous scaling sequence in Chapters 13–17.
- Multimodal serving now includes pooling endpoints; interactive serving now
  includes reasoning streams, tool waits, suspension, and resumption.
- API/grammar ownership was consolidated in Chapter 22, generic stage-
  disaggregation ownership in Chapter 15, and workload methodology in Chapter
  2. Repeated explanations elsewhere were replaced by short, directed links.
- Economics and architecture remain focused in Chapter 25. Multi-tenancy,
  availability abuse, supply-chain trust, retention, governance, and deletion
  evidence now have room in Chapter 26.
- Hardware portability and decision checklists were merged into Appendices B
  and D. The production debugging walkthrough is Appendix I rather than an
  awkward chapter suffix.
- Every chapter exercise has a worked answer in Appendix G; the security and
  adapter exercises now follow the same numbering as the main text.
- Opening visual-map sections were removed. Diagrams and comparison tables now
  sit beside the concept they explain and share one blue visual language,
  caption system, type scale, and responsive renderer.

## Where this book is stronger

The new book treats inference as a stateful scheduling and distributed-systems
problem. Continuous batching, chunked prefill, paged state, prefix reuse,
compilation, precision, speculation, adapters, routing, and disaggregation are
explained as mechanisms that compete for the same latency, memory, and
correctness budgets—not as a list of switches.

Its implementation study is pinned to vLLM and SGLang revisions, and changing
features are supported by primary papers or official documentation. Coverage
extends beyond text generation to pooling, multimodal encoding, diffusion,
reinforcement learning, interactive reasoning, tool suspension, API semantics,
performance science, production operation, economics, security, and deletion
of derived state. The Atlas workload gives the reader one quantitative thread
from a single request to a fleet-level architecture decision.

The rendered edition is also materially stronger than the reference for online
use: it has problem-oriented navigation, responsive diagrams, deep chapter
links, a glossary, source ledger, worked solutions, migration guide, decision
checklists, and an operational debugging playbook.

## Where the reference remains stronger

The reference gives more space to first-principles neural-network instruction,
individual accelerator products, cloud procurement, containers, Kubernetes,
and print-oriented narrative breathing room. This book deliberately assumes
basic neural-network familiarity and keeps rapidly aging product comparisons in
the source layer rather than in the chapter spine.

Its most important remaining limitation is practical evidence. The exercises
are solvable and their answers expose assumptions, but this edition does not
ship a hardware-backed Atlas trace corpus or claim original benchmark results.
A future lab companion should provide pinned deployment manifests, runnable
traces, and repeated measurements on declared accelerator topologies. A future
edition would also benefit from independent technical review of the newer
diffusion, reasoning, and non-NVIDIA portability sections.

## Publication decision

Publish. The final structure now has a clean progression: first request,
service contract, single-engine mechanisms, distributed scaling, new workload
loops, production discipline, and security boundaries. It takes pedagogical
inspiration from the reference while remaining original in thesis, sequence,
terminology, examples, diagrams, exercises, and prose.
