# Revised Backup: Global Dialogue State + Asynchronous Modality State

The previous Backup (“global dialogue SSM + speaker-subsequence SSM”) is
replaced by a falsifiable, pre-training design:

\[
e_t\quad\text{(current fused Student node)}
\]

is the global local dialogue state. For each modality (m\in\{A,T,V\}),
let (mathcal C_m(t)) contain the nearest valid observation before (t),
the nearest valid observation after (t), and (t) itself if observed. With
fixed (lambda>0):

\[
w_{tj}=\exp(-\lambda|j-t|),
\qquad
\bar z_t^m=\frac{\sum_{j\in\mathcal C_m(t)}w_{tj}z_j^m}{\sum_{j\in\mathcal C_m(t)}w_{tj}},
\qquad
\Delta_t^m=\frac{\sum_{j\in\mathcal C_m(t)}w_{tj}(j-t)}{\sum_{j\in\mathcal C_m(t)}w_{tj}}.
\]

The proposed asynchronous state is:

\[
\phi_t^{async}=[e_t;\bar z_t^A,\Delta_t^A;\bar z_t^T,\Delta_t^T;\bar z_t^V,\Delta_t^V].
\]

Before implementing a state-space backbone, run only the frozen-feature ridge
audit:

1. `Local`: (e_t);
2. `Generic`: ([e_t;\operatorname{mean}(e_{\ne t})]) within a conversation;
3. `Asynchronous`: (phi_t^{async});
4. same-modality history shuffle control;
5. same-width random-history control.

Use MOSI rates `0.0, 0.5, 0.7`, seeds `66, 67, 68`, fixed masks, one
`Ridge(alpha=10)` procedure, train-only fitting, and validation-only metrics.
The full SSM is allowed only if asynchronous state improves generic context at
both nonzero rates for at least 2/3 seeds, does not obviously reduce the
complete-input score, loses its gain after history shuffling, and beats the
same-width random-history control. Otherwise the SSM route is closed without
training it.

Current audit outcome: high-missing-rate gains are present, but the complete
condition drops by 0.0307 correlation (below the −0.02 gate), so the SSM
backbone is **not implemented or trained**.
