# PAM-A: Explicit Cross-Modal Episodic Text Memory with Direct Address Supervision

## Scope

PAM-A keeps PAM-E's explicit Text bank, causal OSRAM pre-read, CompletedReadFusion,
read-before-write, and locked CMU-MOSI protocol.  It changes only how the current
context addresses the bank.

PAM-E produced one context query ``q`` and then a closed-form ridge read
``a = (K^T K + lambda I)^-1 K^T q``.  PAM-A instead predicts a signed address
score for every historical episode.

## Explicit bank

For each conversation and every forward call:

```text
B_T(t) = {(k_i^T, v_i^T) | i < t and Text_i is truly observed}

z_i^T = StudentTextProjector(Text_i)
v_i  = z_i^T
k_i  = normalize(K_T(z_i^T))
```

The bank contains T, AT, TV, and ATV episodes.  Predicted Text is never written.

## Current context

At a Text-missing position the context uses only current visible A/V and
observed-only causal OSRAM context:

```text
c_t = normalize(f_context(
    a_mask * z_A,
    v_mask * z_V,
    pattern_embedding(a_mask, v_mask),
    Base_t,
    Gap_Text_t,
))
```

`Base_t` and `Gap_Text_t` come from the same read-only, detached causal OSRAM
pre-read as PAM-E.  Current Text and future utterances never enter ``c_t``.

## Signed episode scores

For every historical episode ``i``, an MLP scorer computes a signed score:

```text
s_{ti} = g(c_t, k_i) in R
a_hat_t = [s_{t1}, ..., s_{tn}]^T
```

There is no softmax.  Negative coefficients are allowed.  The initial scorer
output layer is zero, so the initial read is exactly zero.

The read is:

```text
z_hat_T(t) = V_t a_hat_t
```

where ``V_t=[v_1,...,v_n]`` is the same history Text bank used by the ridge
oracle.

## Direct address supervision

During training the module receives the current Teacher Text target
``z_t^T`` (stop-gradient) and computes the oracle address from the same bank
values:

```text
a_star_t = (V_t^T V_t + lambda I)^-1 V_t^T stopgrad(z_t^T)
```

The first version trains only the address loss, not an additional full-latent
SmoothL1 loss:

```text
L = L_emotion + lambda_a * SmoothL1(a_hat_t, stopgrad(a_star_t))
```

Address loss is computed only on valid Text-missing positions with at least one
historical observed Text episode.  Zero-key padding episodes are excluded.

Configuration: ``key_dim=64``, ``lambda=1e-3``, ``lambda_a=5.0``.  The weight
was chosen after one real batch showed raw address SmoothL1 ~0.0264 and
classification loss ~3.08; with ``lambda_a=5`` the address contribution is
~0.132, large enough to drive addressing without dominating emotion loss.

## Wiring

The predicted ``z_hat_T`` is inserted through the same CompletedReadFusion
path as PAM-E.  OSRAM persistent writes remain real-observed-only.  After the
current classification, a truly observed Text utterance appends its
``(key, value)`` to the bank.  T-only positions append to the bank but do not
receive the address loss themselves.
