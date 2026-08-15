# Research and Originality Policy

This book may be informed by existing books, papers, documentation, source
code, talks, and experiments, but it must remain an independently conceived and
written work.

## Originality rules

1. Do not copy or lightly paraphrase prose, examples, diagrams, exercises, or
   chapter sequences from the reference book or any other source.
2. Develop every explanation from primary evidence and the book's own systems
   model. Cite the evidence that supports factual claims.
3. Create new diagrams, examples, workloads, experiments, and terminology.
4. Use short quotations only when the exact wording is essential, and attribute
   them immediately.
5. Record inspiration separately from manuscript prose so source language does
   not leak into a draft.

## Evidence hierarchy

Prefer evidence in this order:

1. implementation and tests at a recorded commit;
2. primary research papers and specifications;
3. official project and hardware documentation;
4. reproducible measurements produced for this book;
5. maintainers' talks, design discussions, and issue threads;
6. secondary explanations, used mainly to discover primary sources.

Time-sensitive claims must state a date, release, or commit. A repository's
current behavior must never be presented as a timeless property.

## Claim types

Drafts should distinguish three kinds of statements:

- **Principle:** a durable model or design trade-off.
- **Implementation:** how a named revision of a system realizes that principle.
- **Measurement:** a result under an explicitly recorded setup.

This separation prevents an implementation detail from masquerading as a law
and prevents one benchmark result from becoming universal advice.

## Benchmark requirements

Every performance claim should record, where applicable:

- model, precision, quantization, and software revisions;
- accelerator, CPU, memory, interconnect, and topology;
- request arrival process and input/output length distributions;
- concurrency, cache state, warm-up, and failure/retry policy;
- latency percentiles, throughput, goodput, errors, and quality checks;
- exact commands, configuration, raw results, and analysis code.

Comparisons must use equivalent semantics and quality targets. Results that
cannot be reproduced are labeled observations, not conclusions.

## Repository studies

vLLM and SGLang will be studied with the same template:

1. identify the public behavior and user-visible contract;
2. trace the control path from request to scheduler to model runner;
3. trace state ownership and data movement;
4. locate the tests that define expected behavior;
5. reproduce a minimal experiment;
6. explain the trade-off without copying source comments or documentation.

Other engines and runtimes may be included when they reveal a materially
different design. Inclusion is driven by explanatory value, not popularity.

## Review gates

A chapter is ready to publish only when it passes four reviews:

- **Originality:** structure, prose, examples, and figures are independently
  created.
- **Technical:** claims match primary evidence and are versioned when needed.
- **Experimental:** measurements are reproducible and include correctness or
  quality controls.
- **Pedagogical:** the reader can state the decision, trade-off, and failure
  mode after completing the chapter.
