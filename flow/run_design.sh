#!/usr/bin/env bash
# Runs one sky130hd design through the OpenROAD-flow-scripts flow, inside the
# openroad/orfs container, up through routing (so we get both the
# post-placement checkpoint for features and the DRC report for labels).
#
# Usage (host side, via docker run — see flow/docker_run.sh):
#   run_design.sh <design_name> [make_target...]
#
# Expects WORK_HOME to be set (by the caller) to a bind-mounted host dir so
# results/logs/reports/objects survive after the container exits.
set -euo pipefail

DESIGN=$1
shift
TARGETS=${*:-"synth floorplan place cts route finish"}

cd /OpenROAD-flow-scripts/flow
export DESIGN_CONFIG="./designs/sky130hd/${DESIGN}/config.mk"

echo "=== [${DESIGN}] targets: ${TARGETS} ==="
echo "=== [${DESIGN}] WORK_HOME=${WORK_HOME:-.} ==="

# shellcheck disable=SC2086
make ${TARGETS}

echo "=== [${DESIGN}] done. Key artifacts: ==="
ls -la "${WORK_HOME:-.}/results/sky130hd/${DESIGN}/base/" || true
ls -la "${WORK_HOME:-.}/reports/sky130hd/${DESIGN}/base/" || true
