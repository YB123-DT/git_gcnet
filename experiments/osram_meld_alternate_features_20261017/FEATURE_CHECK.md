# Feature-integrity check after low interim MAGTKD scores

INTERNAL DIAGNOSTIC ONLY. No training, inference, dataset mutation or parameter
change was performed for this check. Local CPU NumPy inspection only.

## Direct conversion check

Loaded the hash-pinned canonical ECERC and converted MAGTKD pickle files named
in MAGTKD_PREPARED.json. Loaded the three released source files using prepare.py's
restricted unpickler and magtkd_records(). Joined split/dialogue/utterance IDs,
then compared each selected text/audio_kd/video_kd vector with the converted
text/audio/visual field using np.array_equal.

- All 41,124 vectors matched exactly (13,708 utterances times three streams).
- All MAGTKD feature values were finite across train/dev/test.
- No MAGTKD zero vectors or channels with across-sample variance below1e-12.
- Existing conversion audit verifies canonical ordering,labels,splits and speaker
  selectors; one upstream speaker discrepancy is retained in its provenance.

These checks do not establish that the upstream extractor/checkpoints were
correct or that embeddings generalize. They exclude a narrower family of
conversion corruption,wrong selected-stream copying and numerical-collapse errors.

## Raw feature statistics

Mean utterance L2 norms,computed in float64. These are descriptive,not a quality
score; dimensions and feature spaces differ between releases.

| Source / modality | Dimensions | Train | Dev | Test |
|---|---:|---:|---:|---:|
| ECERC Text |1024|31.759|31.757|31.758|
| ECERC Audio |300|11.055|11.094|11.030|
| ECERC Visual |342|3.432|4.729|4.701|
| MAGTKD Text |768|18.669|18.344|18.346|
| MAGTKD Audio KD |768|11.645|11.623|11.576|
| MAGTKD Visual KD |768|27.117|27.010|27.031|

MAGTKD across-sample variance averaged over channels:

| Stream | Train | Dev | Test |
|---|---:|---:|---:|
| Text |0.315030|0.305081|0.296478|
| Audio KD |0.112128|0.113226|0.109047|
| Visual KD |0.860536|0.773442|0.773591|

The current ModalityProjector has input LayerNorm before its learned projection
and an output LayerNorm. Raw norm differences alone therefore do not establish
that one modality dominates,or that normalization will fix the scores. No new
normalization was added.

## Semantic/protocol boundary

The source audit of official MAGTKD commit
95d0760c26ad2a6ad2181daf372abfd5be348b2d identifies task-supervised cached
features,with audio/video cross-modal distillation and causal dialogue-prefix
text inputs. The chosen streams match its fusion-code convention. These are
not equivalent to three generic independent utterance encoders. Synthetic
history masking cannot remove earlier text already encoded in a cached current
text vector. This limits strict missing-history interpretation independently
of whether performance rises or falls.

The exact training history/checkpoint identity behind the released embeddings
has not been independently reconstructed. Source code alone cannot prove
train/dev/test caches were all generated with the intended same checkpoint.
This remains an upstream-provenance uncertainty,not an observed mismatch.

## Conclusion

Low scores justify investigating the feature pipeline,but do not prove damaged
features. No direct conversion/numerical fault was found in this check. Feature
semantics,upstream generalization and compatibility with the fixed downstream
training recipe remain unresolved. Do not silently switch KD streams,normalise,
tune or restart the authorized fixed-config run on the strength of these stats.
