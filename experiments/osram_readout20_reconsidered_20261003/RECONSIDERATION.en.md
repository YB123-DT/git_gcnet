# Reconsidering substantial Local/Base/Gap blocks

INTERNAL DIAGNOSTIC ONLY — literature/adaptation review, not experimental results.

The previous twenty basic-operator candidates were withdrawn before deployment or training. Their runner and queue now refuse launch. Their implementation remains default-off archival work. No replacement block has been implemented or trained.

Working tree: feature/osram-uniform-forced-text; starting revision5eb6061. Do not silently substitute the historical8f038cb baseline description for current code.

## Revised acceptance test

Preserve a complete computation, not one multiplication, channel scale, or wider MLP. Check both primary methods and actual author code. Parameter count does not establish mechanism substance, and borrowing a larger published block does not establish novelty.

Use only already-computed Local/Base/Gap. Add no OSRAM scan/query/write, temporal cache, completion, label input or auxiliary objective. A residual interface may remain, but must not erase iterative or stateful processing that makes the source method distinct.

Current code confirms that the forward512 read is the flattening of eight actual64-dimensional heads. The five nominal slots are Local/Base/Gap-A/Gap-T/Gap-V, with2–4 active slots because every utterance retains at least one modality. Preserving real head groups produces9/17/25 active inputs including Local. Head indices are not spatial positions, past utterances, objects or known semantic roles. Cross-head coordinate alignment must be specified rather than assumed.

## Provisional structural directions

| Source | Computation to retain | Assessment |
|---|---|---|
| [RRN](https://arxiv.org/abs/1711.08028) | Pair messages, aggregation, shared recurrent node updates, repeated original-input injection | Most direct fit to typed evidence. More iterations do not access more history. Omit original stepwise supervision explicitly. No source license detected; do not copy code |
| [EGT](https://arxiv.org/abs/2108.03348) | Coupled node/edge updates with edge-to-node feedback across multiple layers | Stronger relational state structure, but requires a predeclared role/head graph. Learned edges are not demonstrated reliability |
| [Capsule routing](https://arxiv.org/abs/1710.09829) | Vector votes, competitive assignments, squash and iterative agreement | Conditional. Reset routing each utterance; no margin/reconstruction loss or class-probability interpretation |
| [Slot Attention](https://arxiv.org/abs/2006.15055) | Competitive slot assignment, normalized aggregation, recurrent refinement | Conditional, preferably genuine head tokens. Specify deterministic evaluation; no object-discovery or completion claim |
| [Perceiver IO](https://arxiv.org/abs/2107.14795) | Encode to temporary latent workspace, latent processing, Local-conditioned decode | Secondary. No extra OSRAM query, but tiny input sets remove the original scalability motivation |

These are adaptation judgments, not claims made by the papers about OSRAM and not qualified training entries. Twenty reviewed papers do not mean twenty accepted experiments.

RRN refines node states while recomputing messages. EGT also evolves explicit pair states which affect later node updates. All new states would exist only inside the current readout, not across conversations or utterances.

## Rejected or deferred assumptions

Restormer/XCiT require actual spatial operations; CrossViT requires genuine dual scales. GMN needs two meaningful graphs. MAC and NSM need token-level control or semantic scene-graph structure that these inputs do not supply. Reading-comprehension matching models must not treat arbitrary Local channels as words. HGT/Set Transformer/Universal Transformer remain lower-priority because stripped adaptations can collapse to ordinary attention. NPS has unresolved code and state/rule-mapping gaps.

Four JSON card files and paper_bank.json record authors, canonical links, method/code evidence and uncertainty. Missing official code is explicitly null, not invented.

## Scope and next step

Specify exact within-readout state flow and masking for the strongest RRN/EGT transfers before replacing the experiment manifest. Do not fill a twenty-item list with inapplicable papers, replay the old queue, or infer missing-modality information from readout complexity. No new training commands, scores, or hyperparameter selections are issued here.

Archived correctness/runner suite:41 passed, including the withdrawn-manifest launch guard. An additional legacy suite has21 passes and one pre-existing Relation CUDA exact-equality failure reproduced at5eb6061; current/historical corresponding logits are bit-equal. See the prior screen's VERIFICATION.json. These checks do not validate any proposed replacement architecture or prove performance.
