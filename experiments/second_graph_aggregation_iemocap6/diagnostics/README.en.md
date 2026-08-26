# GPU 4 Retry Diagnostics (English Mirror)

Canonical document: [README.md](README.md).

The retained directories document an infrastructure failure, not completed scientific runs. All seven formal-training attempts assigned to physical GPU 4 exited with code `-9` after 37–59 seconds and 0–5 recorded epochs, without a Python/model traceback. This reproduced with three workers and with one worker on that GPU. The identical locked tasks completed on GPUs 5, 6, or 7.

Failed attempts were preserved before retry, while canonical `fold_5` result directories contain only successful 100-epoch artifacts. Failed attempts are excluded from all metrics and gates. The evidence does not support attributing the failure to either candidate model.

The later uniform three-rate invocation used GPUs 1–7 and added a second bounded diagnostic event: three tasks assigned to GPU 4 exited with code `-9`. Those attempts were moved to diagnostics, excluded from scientific summaries, and the same locked task identities completed successfully after rescheduling on GPUs 1–3. All 27 required new candidate trainings in this uniform layer completed; Original was not retrained. This remains infrastructure evidence and does not change any candidate metric.
