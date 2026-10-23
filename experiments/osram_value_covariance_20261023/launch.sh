#!/usr/bin/env bash
set -euo pipefail

# Launch inside an immutable source snapshot on biggpu. Inference only.
cov_source_sha="${1:?Full immutable source commit required}"
cov_version="${2:?Fresh execution version required}"
cov_root=/data2/yb/remote_experiments/osram_value_covariance_20261023
cov_python=/data2/yb/reproduction_workspace/envs/s0/bin/python
cov_uuid=GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45
cov_output="$cov_root/execution_$cov_version"
if [[ -e "$cov_output" ]]; then
    echo "Refusing to overwrite $cov_output" >&2
    exit 1
fi
mkdir -p "$cov_output"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$cov_python" -m unittest discover -s tests -p 'test_value_covariance*.py' -v \
    > "$cov_root/tests_$cov_version.log" 2>&1
for cov_mode in smoke full; do
    for cov_dataset in CMUMOSI CMUMOSEI; do
        if [[ "$cov_dataset" == CMUMOSI ]]; then
            cov_name=mosi
            cov_reference=/data2/yb/remote_experiments/osram_mosi_mid_lr_20261021/execution_v2/runs/MOSI/seed_66/flat_lr_0.0003
        else
            cov_name=mosei
            cov_reference=/data2/yb/remote_experiments/osram_mosei_gcnet_router_20261020/execution_v1/runs/CMUMOSEI/seed_66/flat
        fi
        cov_args=()
        if [[ "$cov_mode" == smoke ]]; then
            cov_name="smoke_$cov_name"
            cov_args+=(--smoke)
        fi
        "$cov_python" -u -m experiments.osram_value_covariance_20261023.run \
            --dataset "$cov_dataset" --reference "$cov_reference" \
            --gpu-uuid "$cov_uuid" --source-commit "$cov_source_sha" \
            --output "$cov_output/$cov_name" "${cov_args[@]}" \
            > "$cov_root/${cov_name}_$cov_version.log" 2>&1
    done
done
