# OSRAM on IEMOCAP-6

This directory records the complete OSRAM conversational-backbone diagnostic
for IEMOCAP-6. It is an internal Test-oracle diagnostic, not a formal paper
result.

The only model change is the OSRAM H8/Output700 backbone. Observed-set input,
Student Projector, EMA Teacher, MMoE, JEPA objective, cyclic schedule,
features, masks, optimizer, and emotion head remain unchanged. OSRAM uses
eight heads with key/value dimension 32 and exposes the same 500-dimensional
hidden interface as GCNet. The structured missing predictor is retained.

IEMOCAP uses fold 5 (held-out Session 5). Under the inherited `official`
protocol, the held-out session is used for both validation and test, matching
the original GCNet loader topology. Therefore all Test-oracle numbers here are
optimistic diagnostics and must not be presented as independent validation
results.
