# OSRAM on CMU-MOSEI

This directory records the complete OSRAM conversational-backbone diagnostic
for CMU-MOSEI. It is an internal Test-oracle diagnostic, not a formal paper
result.

The only model change is the OSRAM H8/Output700 backbone. The observed-set
input, Student Projector, EMA Teacher, MMoE, JEPA objective, cyclic
missing-rate schedule, frozen features, masks, optimizer, and emotion head
remain unchanged. OSRAM exposes the same 500-dimensional hidden interface as
the GCNet path and retains the structured missing-latent predictor.

CMU-MOSEI uses its official disjoint train/validation/test split (fold 1).
Unlike the inherited IEMOCAP official loader, validation and test are
different partitions here. Nevertheless, the present run intentionally uses
the eight-rate mean Test-oracle checkpoint rule, so the reported numbers are
diagnostic only and must not be presented as formal benchmark results.
