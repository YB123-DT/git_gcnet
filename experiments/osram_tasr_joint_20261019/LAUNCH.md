# One-stage TASR: actual launch

INTERNAL TEST-ORACLE SCREEN. User explicitly requested per-rate Test-best
before any formal job started. No validation-selected job was launched.

- Immutable training source:`c5c26dd1d8dba19ebc816467c9048bbd500b66b7`.
- Parent implementation b3bfe1f was not trained; its unused source snapshot is
  retained without relabeling it as a Test-best run.
- Branch feature/ecerc-features, pushed to YB123-DT/git_gcnet via `github`;
  no push to upstream `origin`.
- Server biggpu, Python `/data2/yb/conda_envs/cmerc-repro/bin/python`,
  Torch1.13.1+cu117/NumPy1.26.4, accepted original MELD feature SHA.
- Source root:
  `/home/qinjinghui/remote_experiments/osram_tasr_joint_20261019/source_c5c26dd`.
- Formal output:
  `/home/qinjinghui/remote_experiments/osram_tasr_joint_20261019/runs_v1/MELD_seed66`.
- Log: experiment-root `train.log`; persistent tmux:`tasr_joint_c5c26dd`.
- Initial worker PID117308. Consult live process identity, not stale PID alone.
- PhysicalGPU1 UUID:`GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`;
  fresh pre-launch free32491MiB/util0%. GPU4 never used, other jobs untouched.
- Home had49GiB available; new checkpoints stay there, not the nearly full data disk.

Exactly one MELD seed66 job,50epochs, batch32, Adam LR1e-5/wd1e-5, constant LR.
From-scratch original network plus F2 TASR, all13,350,218 parameters trainable;
TASR59,423. No pretrained model, no freezing, no second view or extra loss.
Original GCNet random-missing algorithm and original task train_epoch reused.

## Verified real GPU smoke

Full raw record is SMOKE.json, copied from remote smoke_v1/SUMMARY.json.
Five real training batches, five forwards/Adam updates, no JEPA loss. Finished
in12.440seconds including setup/checkpoint recovery; peak allocated1289.304MiB.
Zero-output-initialized TASR logits exactly matched branch-off logits.
Original backbone changed; after five steps all seven audited groups had finite
nonzero task gradients:Encoder, WriteKey, WriteValue, Query, Flat, classifier,TASR.
First step Query/Writer/TASR task gradients were zero due to initial zero output
layers; they are NOT reported as successful nonzero-gradient connections.
Full model/Adam/RNG checkpoint was saved and strictly restored.

CPU verification:62 TASR tests before the final selection change; then29
joint/runner/protocol tests including ACC-only earliest-tie selection. The five
joint tests also passed on the remote runtime. One existing optional torch-sparse
warning (`libcusparse.so.11` unavailable); the new module does not use that op.
Compileall and git diff --check passed.

Remote/local runner SHA256:
`91b59d411efdb1f6bcd7bbbdeaec6773e172ca958f1c79d4d86b5a6fe665f33e`.

## Verified formal start

Fresh process statusrunning, all13,350,218 parameters trainable,
pretrained_model_loaded=false, zero_init_logits_bit_exact=true, no error.
Its INITIAL backbone hash matches the smoke's initial hash, proving the smoke
updates were not carried into formal training. Actual selection identity:
`test-oracle-per-rate-ACC; same-forward UA/WF1; earliest ACC tie`.

Eight full BEST checkpoints with exact selected predictions/masks are retained.
Complete last.pt saves full model/Adam/RNG/history/per-rate selection state.
Each rate's reported UA/W-F1 comes from its ACC-selected predictions, not a
resampled test. Separate final PAIRED_MASK_REPLAY uses common historical masks;
never mix it into Test-best columns or describe eight BEST epochs as one model.
Training is launched, not completed. No final performance claim is made here.
