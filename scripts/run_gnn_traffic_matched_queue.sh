#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 LANE GPU_INDEX" >&2
    exit 2
fi

lane="$1"
gpu_index="$2"
project_dir="/raid/code/dmariaa/spatial-inference-gnn-traffic"
python_bin="/raid/code/dmariaa/spatial-inference/.venv/bin/python"
output_root="$project_dir/output/gnn/dip-matched-traffic"
index_path="/raid/code/dmariaa/spatial-inference/output/gnn/dip-matched-pilot/dip-index.npz"

run_group() {
    local group="$1"
    shift
    local test_sensor_args=()
    local sensor_id
    for sensor_id in "$@"; do
        test_sensor_args+=(--test-sensor-ids "$sensor_id")
    done
    local output="$output_root/group-$group"
    mkdir -p "$output"
    if [[ -f "$output/summary.json" ]]; then
        echo "group-$group already complete"
        return
    fi
    echo "starting group-$group on GPU $gpu_index: $*"
    CUDA_VISIBLE_DEVICES="$gpu_index" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$project_dir/src" \
        "$python_bin" scripts/train_gnn_cached.py \
        --cache cache/metraq/no2-2010-2024 \
        --traffic-cache cache/metraq/traffic-intensity-2015-2024 \
        --output "output/gnn/dip-matched-traffic/group-$group" \
        --architecture sensor-to-grid \
        --local-refinement-layers 1 \
        --time-features \
        --train-start 2015-01-01T00:00 \
        --epochs 100 \
        --batch-size 64 \
        --stride 1 \
        --num-workers 4 \
        --pin-memory \
        --mixed-precision \
        --patience 20 \
        --device cuda \
        "${test_sensor_args[@]}" \
        --test-windows-index "$index_path" \
        --wandb \
        --wandb-project metraq-gnn-test \
        --wandb-entity dmariaa-team \
        --wandb-name "traffic-matched-g$group" \
        --progress > "$output/console.log" 2>&1
    echo "finished group-$group"
}

cd "$project_dir"
case "$lane" in
    0)
        run_group 01 28079024 28079027 28079054 28079058
        run_group 05 28079017 28079039 28079049 28079059
        run_group 09 28079004 28079016 28079036 28079050
        ;;
    1)
        run_group 02 28079018 28079027 28079054 28079058
        run_group 06 28079018 28079039 28079040 28079059
        run_group 10 28079004 28079047 28079048 28079050
        ;;
    2)
        run_group 03 28079024 28079040 28079055 28079060
        run_group 07 28079036 28079038 28079056 28079057
        ;;
    3)
        run_group 04 28079008 28079017 28079055 28079060
        run_group 08 28079016 28079038 28079056 28079057
        ;;
    *)
        echo "unknown lane: $lane" >&2
        exit 2
        ;;
esac
