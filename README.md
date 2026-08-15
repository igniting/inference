# Inference Systems

*Engineering Generative AI from Kernel to Cluster*

This repository contains an original, open technical book about modern
generative-model inference. It studies the ideas embodied in production
systems—including vLLM and SGLang—without reproducing the structure or prose
of any existing book.

Read the published book at **[igniting.github.io/inference](https://igniting.github.io/inference/)**.
The manuscript contains 24 chapters across five parts, plus practical
appendices, a glossary, a source ledger, and a complete
[worked-solutions manual](src/appendices/g-worked-solutions.md). Its research
and originality rules are documented in the
[research policy](src/research-method.md).

Every chapter includes two block diagrams and a compact comparison table. The
diagram source is versioned beside the prose so architectural explanations and
their visuals change together.

## Read locally

The book uses [mdBook](https://rust-lang.github.io/mdBook/).

```bash
mdbook serve --open
```

To verify the static build:

```bash
mdbook build
```

The GitHub Actions workflow publishes every successful `main` build to GitHub
Pages.

## Status

- [x] Define the thesis, audience, and boundaries
- [x] Establish an evidence and originality policy
- [x] Draft all 24 chapters and six appendices
- [x] Compare coverage and pedagogy with the supplied reference book
- [x] Revise the complete manuscript after editorial critique
- [x] Replace open-ended exercises with worked examples and full solutions
- [x] Add a custom professional mdBook theme
- [x] Publish reproducibly through GitHub Pages

The source ledger pins the implementation revisions studied for this edition.
Performance claims should be accompanied by enough workload, hardware, and
software detail to reproduce them.
