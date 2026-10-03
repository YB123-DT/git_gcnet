# Independent optimization-family review

Date: 2026-10-03. Verdict: **CPU APPROVED; no blocking implementation issue found**.
This review did not edit implementations, tests, cards, or integration. No training,
GPU allocation, Git action, new dependency, or source-code import was performed.
The only review-owned artifact is this file.

## Scope and fixed evidence

Reviewed `meaningful_blocks_optimization.py`, its eight tests, `optimization.json`,
the shared tokenizer, and the integration boundary that computes history/Gap masks.
Hashes at review time:

| File | SHA256 |
| --- | --- |
| `gcnet_missing_m3/meaningful_blocks_optimization.py` | `23839aa9471bf068a8f581e460cf064970bbdf985de5fd2e1f1f1c8c78b7e230` |
| `tests/test_meaningful_optimization.py` | `c61fb14a334d9b106a119110e153dd37481d37f52dfef79b83a83f4eabcae822` |
| `experiments/osram_meaningful20_20261003/optimization.json` | `7ad543aecc9fff222bf2d9a7d64d06c078520da12a6f21e0798a520138520571` |
| `gcnet_missing_m3/meaningful_blocks_common.py` | `8647e22cbff8fdf851c72951de0253ed2c419c334e732f029ea6d30406491766` |

The source checks below were against actual pinned author code or publisher code
listings, not only this project's paper cards.

## Mechanism checks

### Hamburger

Local lines 49–66 preserve six detached alternating multiplicative-update steps,
then a differentiable coefficient update and reconstruction. In the pinned author
implementation, `local_inference` is under `no_grad`, while `compute_coef` is called
after it. NMF fixes inverse temperature to one. Local lower/upper breads retain
the source V1 residual/ReLU chain. These match
[author ham.py, lines 44–85 and 210–249](https://github.com/Gsunshine/Enjoy-Hamburger/blob/d9b51f6f197486df68c6e059e396520680157c08/seg/HamNet/hamburger/ham.py)
and [burger.py, lines 17–58](https://github.com/Gsunshine/Enjoy-Hamburger/blob/d9b51f6f197486df68c6e059e396520680157c08/seg/HamNet/hamburger/burger.py),
alongside [paper sections 2.2–2.3](https://zhouchenlin.github.io/Publications/2021-ICLR-Attention.pdf).

Explicit adaptations are appropriate to the approved scope: rank four, immutable
seeded initial basis, equal six-step train/eval inference, tokenwise LayerNorm
instead of batch normalization, no online update, and rolewise difference pooling.
The reconstruction is not detached in its entirety. Inferred factors do not persist.

### CRATE

Local lines 84–94 use one shared subspace projection for Q/K/V, source-style
attention output projection, attention residual followed by sparse PreNorm, and
dictionary ISTA. PyTorch's row-vector orientation gives
`z @ D - (z @ D.T) @ D`; there is no second Transformer residual after ISTA.
This was checked directly against
[author FeedForward, Attention, Transformer, lines 22–95](https://github.com/Ma-Lab-Berkeley/CRATE/blob/674408fa82475fe1f172aa8213e21d4ba608afc4/model/crate.py)
and the [compression/sparsification paper](https://arxiv.org/pdf/2306.01129).
Both complete iterations are retained. Typed head tokens, two 128d layers, and
rolewise difference pooling are declared task adaptations, not image-model reproduction.

### Equilibrium Aggregation

Local lines 106–169 retain residual main/skip mappings, LayerNorm+tanh only on
the first two main branches, final squared-channel mean, positive quadratic
regularizer, whole-energy cardinality scaling, zero initialization, and ten
Nesterov lookahead updates with per-row gradient-threshold masking. This matches
the mechanisms in [publisher supplement C.1–C.5 and Listings 1–2](https://proceedings.mlr.press/v180/bartunov22a/bartunov22a-supp.pdf).
The [main paper](https://proceedings.mlr.press/v180/bartunov22a/bartunov22a.pdf)
provides the optimization-defined aggregate; the local ten-step map is not claimed
to attain an exact equilibrium.

Training preserves the higher-order graph through the inner energy gradients.
Evaluation locally disables inference mode and enables the inner gradient, clones
inference tensors when necessary, detaches iterates, and never calls parameter
`backward`. The task-only omission of the source auxiliary loss and fixed initial
regularizer are already explicit in the card. The publisher listings, rather than
an invented standalone licensed package, remain the reference.

## Fresh verification

Command executed from the worktree:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/yangbin/miniconda3/envs/multimodalerc310/bin/python -m unittest discover -s tests -p test_meaningful_optimization.py -v
```

Result: **8 tests passed in 0.263 s**. Inspected fixtures include source-chain
orientation, NMF forward and truncated backward, complete EA potential/scaling,
quadratic-energy Nesterov recurrence, inactive NaNs, state/RNG invariance, and empty
inputs. These are numerical checks, not performance evidence.

Additional independent inline CPU probe used `torch.manual_seed(197)`, one CPU
thread, Local256, four memory vectors of 8×64, and all seven nonempty modality
availability patterns. It checked training forward/backward, exact inactive-NaN
invariance, zero inactive-input gradients, every parameter's finite gradient,
an SGD step at 0.001, ordinary eval/no_grad/inference_mode agreement, and first-row
versus batched output agreement:

| Candidate | Parameter tensors with finite gradients and an actual update | Result |
| --- | ---: | --- |
| Hamburger | 13 / 13 | PASS |
| CRATE | 24 / 24 | PASS |
| Equilibrium Aggregation | 23 / 23 | PASS |

To avoid validating the implementation only by repeating its forward formula,
the probe also checked two independent derivative oracles:

- For the last NMF coefficient update, frozen factors imply analytic VJP
  `((probe @ B) * (C / (C @ (B.T @ B) + eps))) @ B.T`.
  The implementation matched in float64 with maximum absolute error
  **1.3877787807814457e-16**.
- For the actual learned EA potential and full ten-step map, central differences
  with epsilon 1e-5 matched the outer autodiff gradients below (float64;
  tolerance rtol 2e-4, atol 1e-6). Thus the inner solve is not silently detached.

| EA scalar | Autodiff | Central difference |
| --- | ---: | ---: |
| raw learning rate | -0.05730286380940737 | -0.057302863776254036 |
| raw momentum | -0.021508787856829983 | -0.021508787881319155 |
| raw regularizer | 0.014257511659643668 | 0.014257511656012854 |

## Nonblocking notes and boundaries

- Optimization cores trust the supplied `active` mask; they do not recompute it
  from `availability`. The reviewed integration constructs it from validated
  availability and prior-history existence before calling the core; the tokenizer
  safely masks before projection and again after embeddings. Approval is for this
  integrated contract, not contradictory standalone mask arguments.
- The card's top-level `status` still says research-only/no-implementation. This is
  stale metadata now that code exists; the implementation owner/coordinator can
  update it with the whole experiment manifest. It is not a mathematical blocker.
- GPU higher-order autograd support, peak memory, throughput, real-data stability,
  end-to-end integration, and task accuracy were not established by this review.
  In particular EA's full unroll needs the coordinator's GPU feasibility check.
- No mixed-precision or convergence claim is made. The numerical checks are CPU
  float32/float64 and follow the approved fixed configurations.
