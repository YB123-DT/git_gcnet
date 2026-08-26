# GPU 4 Retry Diagnostics

Chinese: [README.zh.md](README.zh.md). English mirror: [README.en.md](README.en.md).

These directories are infrastructure evidence, not completed scientific runs. During Phase A, all seven formal-training attempts assigned to physical GPU 4 exited with return code `-9` after 37–59 seconds and 0–5 recorded epochs, with no Python/model traceback. The failure reproduced under three-workers-on-GPU-4 and one-worker-on-GPU-4 conditions. The same locked task identities completed successfully on GPUs 5, 6, or 7.

The failed attempts were renamed and preserved before retry. Canonical `fold_5` directories under `results/artifacts/phase_a` contain only successful 100-epoch tasks. No failed attempt contributes to `summary.json` or any reported metric.

This evidence supports the narrow conclusion that GPU 4 was unsuitable for these formal-training processes during this run; it does not imply a GenAgg or Soft Medoid model failure.

The later uniform three-rate invocation used GPUs 1–7 and added a second bounded diagnostic event: three tasks assigned to GPU 4 exited with code `-9`. Those attempts were moved to diagnostics, excluded from scientific summaries, and the same locked task identities completed successfully after rescheduling on GPUs 1–3. Across this uniform layer, all 27 required new candidate trainings completed; Original was not retrained. These observations remain infrastructure evidence and do not change any candidate metric.
