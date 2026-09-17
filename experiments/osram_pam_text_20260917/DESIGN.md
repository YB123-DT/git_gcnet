# PAM-T: Target-Conditioned Predictive Associative Memory (Text target)

## Scope

First version validates a Text target only.  The interface is written so that
Audio/Visual targets can be added later, but this experiment does not implement
or claim arbitrary-modality prediction.

## Two memories

The existing OSRAM memory remains the historical context memory:

- it is updated by real observed slots through the existing Base/Gap/block-write
  semantics;
- it is not modified by PAM-T.

PAM-T adds a second, per-forward dynamic memory:

```text
M_T in R^[B, latent_dim, key_dim]
```

`M_T` is not a parameter.  It is reset to zero for every forward and every
conversation.  The only PAM parameters are:

- the source-condition encoder;
- the scalar write strength `beta`.

The first version deliberately does not use multi-head memory, prototypes,
uncertainty, InfoNCE, or multiple target memories.

## Source condition: no Text leakage

For a Text target, the source-condition encoder sees only the currently visible
A/V information:

```text
q_t = normalize(f_src(
    a_mask * z_A,
    v_mask * z_V,
    embedding(a_mask, v_mask)
))
```

Requirements enforced by the implementation:

- the current Text latent never enters `q_t`;
- complete features never enter `q_t`;
- labels never enter `q_t`;
- future utterances never enter `q_t`;
- OSRAM Base/Gap is not fed to `q_t`; history enters only through `M_T`;
- A-only, V-only and AV use the same encoder and are distinguished by the
  availability mask;
- an utterance with no visible A/V source produces no valid Text query.

Write keys and read queries share the same normalized `q_t`.

## Strict read-before-write

For each valid time step:

1. build `q_t` from visible A/V only;
2. read from the pre-write memory: `z_hat_T_t = M_T^- q_t`;
3. if Text is missing, `z_hat_T_t` may enter the current classification read
   path, and `M_T` is not updated;
4. if Text is observed and at least one of A/V is visible:
   - the current prediction is still produced from `M_T^-` first;
   - then update with the real observed Student Text latent:

```text
old_t = M_T^- q_t
M_T = M_T^- + sigmoid(beta) * (stopgrad(z_T_student) - old_t) * q_t^T
```

5. if Text is missing, complete Text is never used for write; predicted Text is
   never written back;
6. padding, no-A/V-source positions, and invalid utterances neither read nor
   write.

`sigmoid(beta)` is initialized to 0.5.

## How predictions enter classification

PAM-T reuses the existing pre-OSRAM read/write separation:

```text
read_node:
    when Text is missing, predicted Text fills the Text slot and keeps the
    observed/predicted identity;

write_node:
    always the original real observed node.
```

Therefore:

- predicted Text may influence the current Local/query/classification read
  path;
- predicted Text never enters OSRAM write;
- predicted Text never changes availability or turns AV into ATV.

The existing `CompletedReadFusion` and fixed-slot identity code are reused.
Its `active_mask` extension restricts completion correction to Text-missing
positions; the legacy zero-active behavior is unchanged.

This mode does not call the original ContextualM3Predictor / MMoE to predict
Text.

For samples with A/V missing but Text present, the original OSRAM path is kept;
the first version does not predict A/V.

## Training objective

PAM-T uses regression only:

```text
z_T_target = stopgrad(EMA_Text_Projector(complete_text))
L_PAM = SmoothL1(z_hat_T, z_T_target)
```

Prediction mask: every valid training position with at least one visible A/V
source.  Even when Text is present, the prediction is computed read-before-write,
so it remains causal.  The current Text never enters the query; complete Text is
used only for the target loss, never for read or write.

EMA Teacher is only used for the training target.  Memory write values use the
detached Student Text latent, so inference needs only the online Student encoder.

Default PAM loss weight is the previous effective regression coefficient 0.05,
exposed as `--pam-loss-weight`.

## Difference from B2

B2 uses a fixed-parameter source-only predictor followed by one OSRAM pass.
PAM-T's prediction mapping is formed dynamically from real A/V-condition to
Text-latent pairs observed in the current conversation history.

## Configuration

```text
--training-objective pam-text
--completion-path pam-text
--pam-key-dim 64
--pam-loss-weight 0.05
```

The mode is mutually exclusive with the original MMoE joint JEPA,
pretrained-frozen Teacher, predictable subspace, B2 source-only completion,
legacy classification completion, WSC, complete-state JEPA, future-state JEPA,
and other predicted-write paths.

## Files

- `gcnet_missing_m3/pam.py`: PAM-T module.
- `gcnet_missing_m3/b2.py`: optional `active_mask` for `CompletedReadFusion`.
- `gcnet_missing_m3/model.py`: opt-in PAM-T integration.
- `gcnet_missing_m3/train_gcnet.py`: objective, config, loss routing, artifact export.
- `tests/test_pam_text.py`: read/write causality, leakage, padding, gradient and save/load tests.
