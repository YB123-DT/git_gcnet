# OSRAM Query availability ablation

## Question

Does explicitly supplying the current availability vector `a_t` to the OSRAM
Query improve conversational retrieval, beyond the availability signal already
used by Key/Value projections and the hard missing-slot readout?

## A/B definition

| Condition | Query input | Key/Value and readout |
|---|---|---|
| Explicit `a_t` | `[LN(e_t); availability(a_t); speaker; query-type]` | unchanged |
| No explicit `a_t` | `[LN(e_t); 0; speaker; query-type]` | unchanged |

The query-projector width and all learned parameter keys remain identical. The
no-availability condition only zeroes the availability contribution at Query
construction; it does not remove availability from Keys, Values, gap hard
masks, memory writes, or the predictor target mask.

## Hypothesis

Explicit `a_t` may let Base and target-specific Gap Queries specialize to the
current observed subset. If it is redundant with Key conditioning and hard gap
selection, removing it should not materially change Test weighted-F1.
