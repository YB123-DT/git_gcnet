# Local reproduction execution plan

User-authorized scope: seven versions of five baselines, MELD then IEMOCAPFour/Six,
seeds 66/67/68, MR 0.0 through 0.7. Local host user23 only. No remote writes,
SSH, remote execution, uploads, deletion of historical artifacts, or HAMR/OSRAM changes.

1. Audit local worktrees and actual historical artifacts; record incompatibility rather than infer completion.
2. Isolate adapters under this directory, retaining official source versions and patch records.
3. Share dialogue/utterance data, A/T/V order, deterministic evaluation masks, metrics and Stage2-only Test-best ACC.
4. Verify availability/no-target leakage, graph compatibility, both MoMKE causal stages and persisted prediction/checkpoint agreement.
5. Run local GPU smoke checks on UUID whitelist. Freeze per-model configs. Run MoMKE 100/50 vs 200/100 diagnostics without using test scores to choose budget.
6. Queue only verified model/config tasks. Preserve failed logs, do not blindly retry. Save optimizer, epoch, RNG state for true resumption.
7. Aggregate only proven complete folds/seeds, ddof=1. Report incomplete groups and real running PIDs.

The test_best_acc / test_oracle protocol has test-set selection optimism and is not unbiased or leakage-free.

Code ownership: leader runner/dispatcher/config/aggregation/docs; data agent common_data;
independent model agents isolated adapters/{momke,mmin,comp,graph_models}; audit agent audit/legacy_audit.*.
