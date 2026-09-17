# PAM-E: Explicit Cross-Modal Episodic Text Memory

## Scope

PAM-E is the second Text-only PAM version.  It keeps the causal OSRAM
backbone, `osram_write_step=0.6`, mean ObservedSetEncoder, read-before-write,
Flat readout, `CompletedReadFusion`, and the locked MOSI missing-rate protocol.
Only the Text retrieval architecture changes:

- PAM-T compressed history into one associative matrix and applied a delta
  update rule.
- PAM-E stores every prior truly observed Text utterance as an explicit
  episode and retrieves with a learned cross-modal query plus signed ridge
  coefficients.

No A/V target completion, multi-head memory, Transformer, top-k, prototype,
uncertainty, larger query network, new target space, or InfoNCE is used.

## Explicit Text bank

For each conversation and each forward call the bank is rebuilt from zero.
At time `t`:

```text
B_T(t) = {(k_i^T, v_i^T) | i < t and Text_i is truly observed}

z_i^T = StudentTextProjector(Text_i)
v_i  = z_i^T
k_i  = normalize(K_T(z_i^T))
```

`K_T` is an independent Text key encoder:

```text
LayerNorm(256) -> Linear(256, 64) -> GELU -> Linear(64, 64) -> normalize
```

Episodes from patterns `T`, `AT`, `TV`, and `ATV` all enter the bank.  Predicted
Text is never written.

## Cross-modal query

At a Text-missing position the query uses only current truly visible A/V and
observed-only causal OSRAM context:

```text
q_t = normalize(Q(
    a_mask * z_A,
    v_mask * z_V,
    pattern_embedding(a_mask, v_mask),
    Base_t,
    Gap_Text_t,
))
```

`Base_t` and `Gap_Text_t` come from a separate read-only causal OSRAM scan on
the real observed set (`OSRAMBackbone.causal_read_contexts`).  This scan is
read-before-write, uses no future utterance, and its result is detached before
the PAM query encoder.  The query encoder is a LayerNorm plus a compact MLP to
`key_dim=64`; Text never enters it.

## Signed ridge retrieval

For `n` prior observed Text episodes:

```text
K_t = [k_1, ..., k_n] in R^{64 x n}
V_t = [v_1, ..., v_n] in R^{256 x n}
a_t = (K_t^T K_t + 1e-3 I)^(-1) K_t^T q_t
z_hat_T = V_t a_t
```

`torch.linalg.solve` is used, never an explicit inverse.  If a sample has no
historical observed Text, its read is exactly zero and `bank_size=0`.  Coverage
is recorded as `bank_nonempty`.

## Causality and classification wiring

For every valid step:

1. OSRAM causal read on the real observed set produces `Base` / `Gap`.
2. If current Text is missing, PAM-E reads from the bank built from `i < t`.
3. `CompletedReadFusion` places the prediction in the Text slot with
   observed/predicted identity and an active mask that only enables Text-missing
   positions.
4. The completed `read_node` enters the normal OSRAM / Flat classifier.
5. OSRAM persistent writes still use only the original real observed slots.
6. After classification, if current Text is truly observed, append its
   `(key, value)` to the bank.

T-only positions classify with real Text and still append to the bank.
Predicted Text can never write to the bank.

## Training objective

```text
L = L_emotion + 0.05 * SmoothL1(z_hat_T, stopgrad(EMA_Teacher_Text))
```

The PAM loss is computed only on valid Text-missing positions with a visible
A/V source, matching the effective supervision region of the PAM-T full run.
The emotion loss flows through `z_hat_T -> CompletedReadFusion -> classifier`
and trains the query encoder, `K_T`, and upstream student modules.  No InfoNCE
is used.  `key_dim=64`, `lambda=1e-3`, `pam_loss_weight=0.05`.

## Configuration

```text
--training-objective pam-episodic-text
--completion-path pam-episodic-text
--pam-key-dim 64
--pam-loss-weight 0.05
```

PAM-E is mutually exclusive with the original MMoE JEPA, frozen teacher, B2,
legacy classification completion, state JEPA, and other predicted-write paths.
