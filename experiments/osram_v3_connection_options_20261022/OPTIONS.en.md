# Three V3 connection options — research notes, not an approved implementation

Reviewed source: `53d91eab5d39d870b7d1174c3cb505f8c94c7c4b`, `feature/ecerc-features`, 2026-10-22.

The user requests alternatives within V3, not implementation or training. No model, checkpoint, existing experiment, SSH job, GPU inference or training was changed. An existing V3 instance was constructed on CPU solely to count parameters; proposed budgets are algebraic.

## Evidence and invariant core

Keep source tracking, same-source Base–Gap interactions, same-role cross-source interactions, causal joint CHM writes/reads, original queries, Local Skip, task head/loss and GCNet Missing. Sources denote Value-slot contributions, not pure semantic modalities. Numerical examples use standard V3 width64/PairMLP128/Flat1600/CHM8×64, not a silently substituted small-Flat baseline.

Completed MOSI Flat512/V3-128 has three-seed mean W-F1 79.314544 versus large Flat 79.281728: +0.032816 pp overall, −0.339385 pp at high missing. Small-Flat-only seeds67/68 are absent. Completed MOSEI Flat1024 off/on V3-256 changes 83.077816→83.104360, high 79.849565→79.819696. These do not establish robust benefits from widening. See the [MOSI](../osram_mosi_v3_capacity_3seed_20261022/RESULT.md) and [MOSEI](../osram_mosei_v3_capacity_20261022/RESULT.md) archives.

MOSEI's nearly uniform output attention is observed, but not established as a cause of weak performance. [Audit](../osram_v3_source_attention_audit_20261022/RESULT.md). Nested concat exists; its archived 76/100 score is interim, not final150. V3 Direct's local formal result is not complete; its smoke is not a completed experiment. Do not relabel either as a newly discovered method or a proven universal failure.

## A. Preserve final source slots

Keep Layer2 `[T,B,4,3,64]`. Replace the final role queries/output CrossAttention with fixed A/T/V concatenation: each role `192→512`, zero-initialized. Add these four corrections to original reads and retain Flat unchanged. Preserve source/role masks, empty-history/ATV bypasses and existing normalization. Local conditioning remains in PairMLP, not as the removed final query.

This tests the final pooling bottleneck; it does not recover all raw512 information. Source-indexed tokens are already mixed by Layer2, not pure-source attribution. The active-role decoder's difference-space upper bound becomes192 rather than64. More freedom can overfit, and the removed explicit query is a real tradeoff. If useful, a capacity control remains necessary.

Exact V3-slot concatenation is absent from reviewed runs. Nested's five-role320 graph summary concatenated to Flat is a close but distinct precedent; concatenation itself is not novel.

## B. Independent V3 features inside Flat

Keep current V3 through four role-conditioned64 outputs; concatenate them into z256. Remove four `64→512` decoders and leave raw Base/Gap unmodified. For x4352:

`p = W1 LN_x(x) + b1 + P MaskedLN_z(z)`

`h = LN_out(Skip(L) + W2 Dropout(GELU(p)) + b2)`

P is bias-free `256→1600`. Keep original LN4352, and use separate LN256 followed by role masking and a full no-history/ATV/padding bypass so learned affine terms cannot create inactive evidence. LN256 still couples role statistics and changes amplitude information; this is part of the proposed interface, not a neutral detail. This enters before the original GELU, unlike the old global residual after the whole Flat adapter.

This is mathematically a linear layer over separately normalized concatenated inputs, not a distinct new mechanism from such concatenation. It gives relation features separate coefficients rather than requiring reconstruction in the original read coordinates.

For from-scratch training, P has ordinary nonzero initialization and original W2/b2 remain zero. Initial predictions can match the original Local Skip path, but the first W2 gradient differs. One extra zero decoder barrier disappears; the original W2 barrier remains. P's initial influence is not necessarily small, and adding before GELU changes the old term's nonlinear response. This is not proof of better long-run optimization. It does not guarantee identity when loading a pretrained nonzero W2.

Nested concat is a close precedent but uses another representation and enlarged joint input LN. This option keeps original LN and supplies a separate V3 entrance. It retains the final64-per-role bottleneck, may still be ignored, and must not be combined with A in a first comparison.

## C. Preserve original Memory Head coordinates

Represent source reads as `[T,B,role4,source3,head8,value64]`. Each CHM head has its own bias-free64→64 projection, shared across sources and roles within that head. Execute V3's two relationship layers and final role query independently per head. Share PairMLP, attention and norms across heads; do not instantiate eight entire V3 blocks.

Add initially zero head embeddings8×64 to the existing Local64 conditions and role queries, keeping PairMLP input343 without unit-variance embedding perturbations to query scale. Use32 role/head-specific zero-initialized64→64 decoders; concatenate eight outputs per role back to512 and add original reads. Current masks and norm policy remain.

The eight heads are CHM heads, not the four latent MHA heads. Projection and decoder budgets equal current V3 exactly; head embeddings add512 parameters. This removes the single global64-dimensional correction exit but forbids unrestricted cross-head input mixing. Relationship/attention token batches grow about8×, not a measured8× total runtime. Semantic head specialization has not been proven.

Nested/Local8 already modeled original heads. The untested combination here is the explicit source axis plus per-head V3 hierarchy, not head modeling itself.

## Budgets and recommendation

| Design | Block parameters | Status |
|---|---:|---|
| Current V3 | 335,006 | Counted from existing class |
| A | 514,718 | Algebraic proposal |
| B | 611,998 | Algebraic proposal, bias-free P and LN256 |
| C | 335,518 | Algebraic proposal, head-specific projections/decoders and shared core |

Priority by structural value and parameter control: **C, then A, then B**. C tests native-head preservation at nearly equal parameter count, but not equal compute. A is the most local and cheaper output-interface comparison and can come first if cost is the priority. B probes a separate entrance but is closest to a previously attempted concatenation route. These are three variants of V3, not three validated new methods. Do not run them until the user selects a design.

[MulT's pinned author implementation](https://github.com/yaohungt/Multimodal-Transformer/blob/a670936824ee722c8494fd98d204977a1d663c7a/src/models.py#L101) concatenates directional outputs before temporal encoding and modality outputs later. [DialogueGCN](https://aclanthology.org/D19-1015.pdf) and the [prior connection review](../osram_post_temporal_connection_review_20261022/RESULT.en.md) provide additional separate-channel examples. They motivate engineering hypotheses, not OSRAM performance guarantees or causal proof that concatenation caused published gains. C is primarily grounded in current CHM/Nested code.

Any later screen should preserve the chosen baseline/protocol and report paired task metrics, then necessary capacity/on-off controls if promising. No automatic multi-seed expansion, new losses, removal of all norms, compound A/B/C or new diagnostic round is approved. This note verifies provenance/dimensions only, not proposed implementations or live remote progress.
