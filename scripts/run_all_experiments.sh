#!/usr/bin/env bash
# Run evaluation for every experiment config in experiments/configs/
# Usage: ./scripts/run_all_experiments.sh [--configs-dir DIR] [--num-questions N] [--dry-run]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_SCRIPT="$PROJECT_ROOT/experiments/scripts/run_evaluation.py"
CONFIGS_DIR="$PROJECT_ROOT/experiments/configs"
LOG_DIR="$PROJECT_ROOT/results/logs"
NUM_QUESTIONS=""
DRY_RUN=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --configs-dir) CONFIGS_DIR="$2"; shift 2 ;;
        --num-questions) NUM_QUESTIONS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# Collect all experiment YAMLs, excluding baseline and template
CONFIGS=()
while IFS= read -r f; do
    CONFIGS+=("$f")
done < <(
    find "$CONFIGS_DIR" -name "*.yaml" \
        ! -name "baseline.yaml" \
        ! -name "template.yaml" \
    | sort
)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "No experiment configs found in $CONFIGS_DIR"
    exit 1
fi

echo "============================================"
echo "Running ${#CONFIGS[@]} experiments"
echo "Project root: $PROJECT_ROOT"
echo "Log dir:      $LOG_DIR"
[[ -n "$NUM_QUESTIONS" ]] && echo "Questions:    $NUM_QUESTIONS (limited)"
$DRY_RUN && echo "DRY RUN — commands will be printed but not executed"
echo "============================================"
echo ""

PASSED=0
FAILED=0
SKIPPED=0
FAILED_CONFIGS=()

for CONFIG in "${CONFIGS[@]}"; do
    EXP_NAME=$(basename "$CONFIG" .yaml)
    LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

    # exp3 configs need --artifacts (they have experiment-specific indices)
    ARTIFACTS_FLAG=""
    if [[ "$CONFIG" == */exp3/* ]]; then
        ARTIFACTS_FLAG="--artifacts"
    fi

    CMD=(
        python3 "$EVAL_SCRIPT"
        --config "$CONFIG"
        $ARTIFACTS_FLAG
    )
    [[ -n "$NUM_QUESTIONS" ]] && CMD+=(--num-questions "$NUM_QUESTIONS")

    echo ">>> $EXP_NAME"
    echo "    config:  $CONFIG"
    echo "    log:     $LOG_FILE"
    [[ -n "$ARTIFACTS_FLAG" ]] && echo "    index:   experiment-specific"

    if $DRY_RUN; then
        echo "    cmd:     ${CMD[*]}"
        echo ""
        (( SKIPPED++ ))
        continue
    fi

    START=$(date +%s)

    if (cd "$PROJECT_ROOT" && "${CMD[@]}" > "$LOG_FILE" 2>&1); then
        END=$(date +%s)
        echo "    status:  OK ($(( END - START ))s)"
        (( PASSED++ ))
    else
        END=$(date +%s)
        echo "    status:  FAILED ($(( END - START ))s) — see $LOG_FILE"
        FAILED_CONFIGS+=("$EXP_NAME")
        (( FAILED++ ))
    fi
    echo ""
done

echo "============================================"
echo "Done: $PASSED passed, $FAILED failed, $SKIPPED skipped"
if [[ ${#FAILED_CONFIGS[@]} -gt 0 ]]; then
    echo "Failed:"
    for f in "${FAILED_CONFIGS[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
