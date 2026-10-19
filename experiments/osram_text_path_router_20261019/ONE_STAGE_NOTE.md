# What a genuine one-stage version would require

User follow-up: "可能要一阶段训练". This is a proposed next experiment, not evidence
that freezing caused the failed/unchanged pilot, and not an already-started job.

A genuine one-stage version starts the original Encoder/OSRAM/Flat/head and Router
from random initialization and trains jointly. Loading the protected trained Flat
and unfreezing it is fine-tuning, not from-scratch one-stage training.

Current code is intentionally Router-only; simply changing requires_grad is
insufficient:

1. `candidates.py::pack_view/candidate_readouts` use no_grad and detached captured
   inputs. One-stage needs a separate opt-in graph-connected path, while frozen
   behavior remains unchanged.
2. `router.py::forward` detaches features AND candidate logits. At minimum the
   three candidate probability paths must remain connected for end-to-end task
   supervision. Router-feature gradients should be an explicit documented choice,
   not an accidental detach change.
3. `JointSourceTracker.read_base` is an audit-only no_grad/detached computation.
   A trainable source-use path needs a differentiable Base source read. The same
   original joint write terms and queries must remain intact, not three new writes.
4. Original Flat has Dropout. All three candidate readouts must reuse the same
   Dropout realization; otherwise they differ in both evidence and random masks.
   Preserve the post-full-forward RNG stream so added readouts do not change
   subsequent original random-missing behavior. No second Memory scan is needed.
5. Keep the trigger, original missing protocol and original joint write unchanged;
   use only the probability-mixture task CE, no auxiliary expert loss initially.
   Preserve a matched from-scratch Flat control and exact mask provenance.

Potential one-stage advantage: representations and decision paths can co-adapt to
the mixture objective. Risk: paths may co-adapt to ignore the removed source or
Router may still favor the full candidate; gain is not guaranteed. This remains
the same limited current-missing/past-seen Text scope and cannot improve fully
Persistent Text absence without changing the research question.
