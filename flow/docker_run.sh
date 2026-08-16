#!/usr/bin/env bash
# Host-side wrapper: runs flow/run_design.sh inside the openroad/orfs container
# with results persisted to ./flow_work/<design>/ on the host.
#
# Usage: ./flow/docker_run.sh <design> [make_target...]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESIGN=$1
shift
TARGETS="${*:-}"

mkdir -p "${PROJECT_DIR}/flow_work/${DESIGN}"

MSYS_NO_PATHCONV=1 docker run --rm \
  -v "${PROJECT_DIR}:/workspace" \
  -e WORK_HOME="/workspace/flow_work/${DESIGN}" \
  openroad/orfs:latest \
  bash /workspace/flow/run_design.sh "${DESIGN}" ${TARGETS}
