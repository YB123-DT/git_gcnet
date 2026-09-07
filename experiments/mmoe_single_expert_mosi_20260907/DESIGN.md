# MMoE routing weak-point diagnostic

## Question

Does expert competition/routing in the six-direction Missing-M3 predictor
help the current OSRAM model, or is it an avoidable source of noise and
overfitting?

## Minimal contrast

| Condition | Expert configuration | Other components |
|---|---:|---|
| Full reference | 4 experts, top-2 | inherited from H8/32/32/700 OSRAM |
| Single-expert control | 1 expert, top-1 | identical |

The single-expert condition keeps the same predictor API, source/target
embeddings, two gates, target heads, JEPA objective, and optimizer. It removes
expert competition and specialization without introducing a new predictor
architecture. The reference is inherited and is not retrained.

This is a mechanism diagnosis, not a parameter-matched final comparison: the
single-expert control has fewer expert parameters. If it is worse, routing is
likely useful; if it is equal or better, routing is a likely weak point and a
later parameter-matched shared-MLP control may be justified.

## Locked protocol

- Dataset: CMU-MOSI, fold 1
- Seeds: 66, 67, 68, 69, 70
- Missing rates: 0.0 through 0.7
- Training schedule: cyclic mixed-rate
- Selection: one checkpoint per seed by the eight-rate Test-oracle mean;
  the per-rate maxima are retained only as a diagnostic view
- Backbone: OSRAM H8, key/value 32, output 700
- Unchanged: masks, Student Projectors, EMA Teacher, JEPA loss, OSRAM,
  classification head, learning rate, batch size, optimizer and feature files

## Scope

No loss, routing formula, expert gate, backbone, mask schedule, or training
protocol is changed in this diagnostic.
