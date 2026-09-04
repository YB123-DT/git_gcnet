# CMU-MOSI Slot Fusion Localization

## Question

Determine whether Slot Missing-M3 fails to exploit Audio and Visual because
the modalities contain no sample-level signal, because the Slot fusion
disturbs the dominant Text representation, or because GCNet removes useful
non-text information.

## Protocol

- Dataset: CMU-MOSI official test split.
- Model: existing validation-selected Slot Missing-M3 checkpoints.
- Seeds: 66--70.
- Training performed: none.
- Fixed inputs: `A`, `T`, `V`, `AT`, `AV`, `TV`, and `ATV` on the same test
  utterances.
- Correspondence controls: shuffle Audio and/or Visual between valid test
  utterances while preserving the availability pattern and every other
  input.
- Captured stages: Slot output, pre-graph BiLSTM, temporal branch, speaker
  branch, final hidden, and prediction.
- The diagnostic never selects a checkpoint, epoch, or hyperparameter from
  test results.

The remote inference environment uses PyG 2.7, whereas the checkpoints use
the earlier `GraphConv` state names. The loader performs the exact semantic
rename `lin_l -> lin_rel` and `lin_r -> lin_root`, checks every tensor shape,
and then requires a strict state load. No parameter is dropped or randomly
initialized.

## Fixed-pattern result

| Input | Five-seed test W-F1 (%) |
|---|---:|
| A | 50.43 |
| T | **85.93** |
| V | 49.39 |
| AT | 85.43 |
| AV | 55.48 |
| TV | 85.76 |
| ATV | 85.71 |

Adding a weak modality to Text does not improve the mean:

| Contrast | W-F1 delta (points) | Positive seeds |
|---|---:|---:|
| AT - T | -0.50 | 2/5 |
| TV - T | -0.18 | 2/5 |
| ATV - T | -0.22 | 2/5 |

This rules out the claim that the current fusion consistently extracts a
positive Audio/Visual increment over Text.

## Correct correspondence versus shuffled modalities

| Contrast | Structural cost relative to T | Correct-pair recovery | Net versus T |
|---|---:|---:|---:|
| shuffled-A T -> correct AT | -0.84 | +0.35 | -0.50 |
| T shuffled-V -> correct TV | -0.71 | +0.54 | -0.18 |
| T shuffled-A/V -> correct ATV | -0.95 | **+0.73** | -0.22 |

The joint correct-pair recovery is positive for all five seeds. Therefore
Audio and Visual are not completely ignored and are not devoid of
utterance-specific information. The failure is that activating their Slot
and changing the pattern imposes a larger representational cost than the
model recovers from correct cross-modal correspondence.

## Layer localization

Mean normalized representation changes relative to Text-only input:

| Added evidence | Slot | Pre-graph BiLSTM | Temporal | Speaker | Final |
|---|---:|---:|---:|---:|---:|
| Audio | 0.405 | 0.235 | 0.124 | 0.121 | 0.120 |
| Visual | 0.502 | 0.300 | 0.178 | 0.136 | 0.153 |
| Audio + Visual | 0.547 | 0.328 | 0.202 | 0.156 | 0.174 |

Correct-versus-shuffled sample-level changes are smaller but nonzero:

| Correspondence | Slot | Pre-graph BiLSTM | Final | W-F1 recovery |
|---|---:|---:|---:|---:|
| Audio in AT | 0.144 | 0.095 | 0.067 | +0.35 |
| Visual in TV | 0.226 | 0.207 | 0.148 | +0.54 |
| Audio + Visual in ATV | 0.226 | 0.202 | 0.149 | +0.73 |

About 68--70% of the raw Slot perturbation magnitude is attenuated by the
time the representation reaches the final hidden. Both the pre-graph BiLSTM
and graph path contribute to this compression. This measurement alone does
not label attenuation as harmful: it removes nuisance variation and useful
correspondence together. Crucially, the negative net fusion result already
starts from a Slot representation that changes strongly when weak modalities
are activated.

## Diagnosis

The primary observed failure is **quality-uncalibrated, non-anchor-preserving
fusion**, not absence of Audio/Visual signal and not proven failure of the
GCNet topology.

1. Text is dramatically stronger than Audio or Visual on MOSI (`85.93` versus
   roughly `50`).
2. Slot fusion applies one shared `LayerNorm -> Linear -> GELU` to the three
   modality slots plus a pattern embedding. It has no invariant path that
   preserves the strong observed Text representation when another modality
   becomes available.
3. Activating Audio/Visual changes the Slot representation by `40--55%`, but
   correct sample correspondence recovers only `0.35--0.73` W-F1 points.
4. GCNet compresses the perturbation and does not restore a positive net
   gain. Prior SDT/SDR backbone replacements also failed to improve MOSI, so
   the evidence does not support replacing the graph core again.

The falsified explanation is "Audio/Visual are simply ignored." The supported
explanation is: **weak non-text evidence is injected too symmetrically and
disturbs the dominant Text representation; the useful aligned component is
real but smaller than this fusion cost.**

## Consequence for the next model

The next intervention must be isolated to node construction. When Text is
observed, it should preserve a direct Text anchor and admit Audio/Visual only
as bounded, sample-conditioned residual evidence. When Text is missing, the
same module must fall back to an Audio/Visual anchor without requiring a
hallucinated Text slot. GCNet, JEPA, masks, optimizer, and readout should stay
unchanged for the causal A/B.

Raw machine-readable evidence is stored in `results.json`.
