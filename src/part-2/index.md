# Part II — Inside a Single Engine

One request, one engine, opened end to end. A scheduler chooses the next unit
of work, a memory manager finds room for its state, kernels move the bytes,
compiled graphs remove the launch cost, and the numerical and decoding choices
determine what the engine may safely trade away. Chapter 12 extends the
single-engine story to multi-tenant adapter serving, where many customized
models share one set of base weights.

**Chapters 5–12**
