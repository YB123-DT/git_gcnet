#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${GCNET_REMOTE_HOST:-biggpu}"
REMOTE_ROOT="${GCNET_REMOTE_ROOT:-/data2/yb/paper/GCNet_TPAMI_single_view_dev}"
TEST_PY="${GCNET_TEST_PY:-/data2/yb/reproduction_envs/s0/bin/python}"
TRAIN_PY="${GCNET_TRAIN_PY:-/data2/yb/reproduction_envs/gcnet-official/bin/python}"

usage() {
    cat <<'EOF'
Usage:
  scripts/remote_missing_m3.sh preflight
  scripts/remote_missing_m3.sh sync RELATIVE_PATH [RELATIVE_PATH ...]
  scripts/remote_missing_m3.sh test PYTEST_ARGS...
  scripts/remote_missing_m3.sh train GPU PYTHON_ARGS...

Examples:
  scripts/remote_missing_m3.sh sync gcnet_missing_m3/model.py tests/test_missing_m3.py
  scripts/remote_missing_m3.sh test tests/test_missing_m3.py -k paper_faithful
  scripts/remote_missing_m3.sh train 0 -m gcnet_missing_m3.train_gcnet --help
EOF
}

shell_join() {
    printf '%q ' "$@"
}

remote_run() {
    local command
    command="cd $(printf '%q' "$REMOTE_ROOT") && $(shell_join "$@")"
    ssh "$REMOTE_HOST" "$command"
}

preflight() {
    ssh "$REMOTE_HOST" "set -eu
test -d $(printf '%q' "$REMOTE_ROOT")
test -x $(printf '%q' "$TEST_PY")
test -x $(printf '%q' "$TRAIN_PY")
cd $(printf '%q' "$REMOTE_ROOT")
$(printf '%q' "$TEST_PY") -c 'import pytest, torch, torch_geometric'
$(printf '%q' "$TRAIN_PY") -c 'import numpy, sklearn, torch, torch_geometric'
printf '%s\n' 'remote-missing-m3 preflight: OK'"
}

sync_paths() {
    if (( $# == 0 )); then
        echo "sync requires at least one repository-relative path" >&2
        exit 2
    fi
    local path parent
    for path in "$@"; do
        if [[ "$path" = /* || "$path" == *".."* || ! -e "$path" ]]; then
            echo "refusing unsafe or missing sync path: $path" >&2
            exit 2
        fi
        parent="$(dirname "$path")"
        ssh "$REMOTE_HOST" "mkdir -p $(printf '%q' "$REMOTE_ROOT/$parent")"
        rsync -a "$path" "$REMOTE_HOST:$REMOTE_ROOT/$path"
    done
}

command="${1:-}"
if [[ -z "$command" ]]; then
    usage
    exit 2
fi
shift

case "$command" in
    preflight)
        if (( $# != 0 )); then
            usage
            exit 2
        fi
        preflight
        ;;
    sync)
        sync_paths "$@"
        ;;
    test)
        if (( $# == 0 )); then
            echo "test requires pytest arguments" >&2
            exit 2
        fi
        remote_run "$TEST_PY" -m pytest "$@"
        ;;
    train)
        if (( $# < 2 )); then
            echo "train requires GPU and Python arguments" >&2
            exit 2
        fi
        gpu="$1"
        shift
        if [[ ! "$gpu" =~ ^[0-9]+$ || "$gpu" == "4" ]]; then
            echo "GPU must be a numeric non-4 device index" >&2
            exit 2
        fi
        remote_run env "CUDA_VISIBLE_DEVICES=$gpu" "$TRAIN_PY" "$@"
        ;;
    *)
        usage
        exit 2
        ;;
esac

