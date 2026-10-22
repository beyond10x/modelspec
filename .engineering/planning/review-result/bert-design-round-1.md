---
format: aep.planning-md/3
id: review-result:bert-design-round-1
kind: review-result
status: active
title: Plan critic design, epic:bert-from-scratch, round 1
relations:
- reviews: epic:bert-from-scratch
- reviews: story:bert-adamw-warmup
- reviews: story:bert-attention
- reviews: story:bert-corpus-splits
- reviews: story:bert-dense-gelu
- reviews: story:bert-dropout
- reviews: story:bert-embedding-lookup
- reviews: story:bert-heldout-evaluation
- reviews: story:bert-layer-norm-softmax
- reviews: story:bert-mlm-batches
- reviews: story:bert-mlm-cross-entropy
- reviews: story:bert-tiny-cpu-model
- reviews: story:bert-tiny-gpu-pretraining
- reviews: story:bert-tokenizer
- reviews: story:bert-train-data-binding
- reviews: story:bert-vulkan-kernels-attention-loss-adamw
- reviews: story:bert-vulkan-kernels-rowwise
revision: 1
---
needs-revision

1. `story:bert-train-data-binding` — its Scope says it adds the run record's type in `spec/domains/`, but `story:bert-mlm-batches` already types that record in its second settlement, and the body never says which story owns the type — `.engineering/planning/story/bert-train-data-binding.md:64`, `.engineering/planning/story/bert-mlm-batches.md:38-43`

   - mlm-batches line 43 gives binding only the writing: "`story:bert-train-data-binding` writes the record where this settlement puts it". Its acceptance scenario 1 requires typed declarations in `data.yaml`.
   - The AGENTS.md "Planning and specification" section requires the type to exist before code uses it.
   - Closing the seam: binding line 64 should read "the writer only; the type is declared by the `story:bert-mlm-batches` settlement", or the type should move to binding with a validating scenario.

Everything else holds: no cycle, no split abstraction, no horizontal slice (each operator story is demonstrated alone by its own finite-difference test), no unrecorded read of state a sibling produces.

What I read: all 16 `story:bert-*` bodies and `epic:bert-from-scratch`; `aep plan artifact list`, `relations`, `graph`, `validate` (valid) and `waves`. 33 `depends_on` edges walked, including the one to `epic:m1-vulkan-execution`; no edge from the observatory stories or the two epics leads back into a `bert-*` story. About 20 acceptance reads traced to a producer through a direct or transitive `depends_on` path.

What I could not establish:
- 14 of the 16 stories form one queue; every edge has its reason in an "Order on shared files" section (`ops.rs`, `resolve.rs`, `autodiff.rs`, `DESIGN.md`, `CHANGELOG.md`, `runner.rs`). Splitting those files per operator was not judged.
- Out of lane (acceptance): scenario 4 of `story:bert-tiny-cpu-model` reads "the fixture packed data", which `story:bert-train-data-binding` builds only in a temporary directory under `target/`; no story persists it.
- Out of lane (scope): `story:bert-corpus-splits` and `story:bert-mlm-batches` each may write the `Dataset` record, depending on an open settlement; each body handles both outcomes.

```findings
- file: .engineering/planning/story/bert-train-data-binding.md
  line: 64
  category: design
  severity: warning
  verdict: needs-revision
  origin: introduced
  message: "its Scope says it adds the run record's type in spec/domains/ while story:bert-mlm-batches already types that record (bert-mlm-batches.md:38-43), and the body does not say which story owns the type"
```
