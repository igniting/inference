# Inference Systems

*Engineering Generative AI from Kernel to Cluster*

This repository contains an original, open technical book about modern
generative-model inference. It studies the ideas embodied in production
systems—including vLLM and SGLang—without reproducing the structure or prose
of any existing book.

The project is currently in the **outline review** stage. Start with the
[complete chapter outline](src/outline.md) and the
[research and originality policy](src/research-method.md).

## Read locally

The book uses [mdBook](https://rust-lang.github.io/mdBook/).

```bash
mdbook serve --open
```

To verify the static build:

```bash
mdbook build
```

The GitHub Actions workflow publishes the generated site to GitHub Pages after
Pages is configured to use **GitHub Actions** as its source.

## Status

- [x] Define the book's thesis, audience, and boundaries
- [x] Produce the full part, chapter, and section architecture
- [x] Establish an evidence and originality policy
- [ ] Review and approve the outline
- [ ] Create chapter briefs and assign evidence
- [ ] Draft, review, benchmark, and illustrate the chapters

No content license has been selected yet. Contributions should wait until the
copyright and licensing policy is decided.
