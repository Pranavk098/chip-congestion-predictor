#!/usr/bin/env bash
# End-to-end pipeline: OpenROAD flow -> feature/label extraction -> dataset -> train -> evaluate.
#
# Prereqs: Docker running, `docker pull openroad/orfs:latest` done, Python venv
# at .venv with requirements.txt installed.
#
# Produces a single train/test split (train=gcd,aes,jpeg,riscv32i test=ibex).
# For the full leave-one-design-out study across all 5 designs, use
# scripts/cross_validate.py instead (what the README's numbers come from).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

DESIGNS=("gcd" "aes" "ibex" "jpeg" "riscv32i")
TEST_DESIGN="ibex"
PY=".venv/Scripts/python.exe"
BIN_SIZE_UM=2.0

mkdir -p dataset/raw_v2 outputs

# 1. Run each design through the OpenROAD flow (synth -> floorplan -> place -> cts -> route -> finish)
for d in "${DESIGNS[@]}"; do
  if [ ! -f "flow_work/${d}/results/sky130hd/${d}/base/3_3_place_gp.odb" ]; then
    echo "=== Running flow for ${d} ==="
    bash flow/docker_run.sh "$d" "synth floorplan place cts route finish"
  else
    echo "=== ${d}: flow artifacts already present, skipping ==="
  fi
done

# 2. Extract features (9-channel: cell/pin density, RUDY family, macro_region,
#    real GR congestion) and labels (DRC report -> raster).
#
# TritonRoute iterates and typically converges to a clean (0-violation) final
# report on these small/medium sky130hd designs, which would leave us with an
# all-negative label set. We instead label from its iteration-5 intermediate
# report when it's non-empty: those are real violations the router had to
# spend extra effort fixing -- i.e. genuine routing hotspots. Falls back to
# the final report if no intermediate report exists (e.g. gcd, clean at
# every iteration). Same logic for the congestion report: only used when the
# design actually triggered a global-routing congestion violation.
for d in "${DESIGNS[@]}"; do
  ODB="flow_work/${d}/results/sky130hd/${d}/base/3_3_place_gp.odb"
  REPORTS_DIR="flow_work/${d}/reports/sky130hd/${d}/base"
  DRC_RPT="${REPORTS_DIR}/5_route_drc.rpt"
  if [ -s "${REPORTS_DIR}/5_route_drc.rpt-5.rpt" ]; then
    DRC_RPT="${REPORTS_DIR}/5_route_drc.rpt-5.rpt"
  fi
  CONG_RPT="${REPORTS_DIR}/congestion-5.rpt"
  DUMP_JSON="flow_work/${d}/layout_dump_v2.json"
  echo "=== [$d] DRC=${DRC_RPT}  CONG=$([ -f "$CONG_RPT" ] && echo "$CONG_RPT" || echo none) ==="

  MSYS_NO_PATHCONV=1 docker run --rm \
    -v "${PROJECT_DIR}:/workspace" \
    -e ODB_IN="/workspace/${ODB}" \
    -e JSON_OUT="/workspace/${DUMP_JSON}" \
    openroad/orfs:latest \
    bash -c "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad -exit -no_init /workspace/feature_extraction/dump_layout.tcl"

  if [ -f "$CONG_RPT" ]; then
    (cd feature_extraction && "../${PY}" rasterize.py "../${DUMP_JSON}" "../dataset/raw_v2/${d}_features.npz" --bin-size-um "$BIN_SIZE_UM" --congestion-rpt "../${CONG_RPT}")
  else
    (cd feature_extraction && "../${PY}" rasterize.py "../${DUMP_JSON}" "../dataset/raw_v2/${d}_features.npz" --bin-size-um "$BIN_SIZE_UM")
  fi

  (cd feature_extraction && "../${PY}" extract_labels.py "../${DRC_RPT}" "../${DUMP_JSON}" "../dataset/raw_v2/${d}_labels.npz" --bin-size-um "$BIN_SIZE_UM")
done

# 3. Build patch dataset: train on everything except TEST_DESIGN
TRAIN_DESIGNS=()
for d in "${DESIGNS[@]}"; do
  [ "$d" != "$TEST_DESIGN" ] && TRAIN_DESIGNS+=("$d")
done
echo "=== Building dataset (train=${TRAIN_DESIGNS[*]}  test=${TEST_DESIGN}) ==="
"$PY" dataset/build_dataset.py --train-designs "${TRAIN_DESIGNS[@]}" --test-designs "$TEST_DESIGN" \
  --patch-size 32 --stride 16 --out-dir dataset --raw-dir dataset/raw_v2

# 4. Train (Attention U-Net by default; pass --arch unet for the plain-U-Net baseline)
echo "=== Training ==="
"$PY" model/train.py --dataset-dir dataset --out-dir outputs --epochs 100

# 5. Evaluate
echo "=== Evaluating on held-out test design (${TEST_DESIGN}) ==="
"$PY" model/evaluate.py --dataset-dir dataset --checkpoint outputs/best_model.pt --out-dir outputs --split test
"$PY" model/evaluate.py --dataset-dir dataset --checkpoint outputs/best_model.pt --out-dir outputs --split val

echo "=== Done. See outputs/eval_test.json, outputs/predictions_test.png ==="
