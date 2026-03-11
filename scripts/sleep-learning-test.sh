#!/usr/bin/env bash
# =============================================================================
# sleep-learning-test.sh — Accelerated sleep / learning pipeline test
#
# Seeds synthetic episodic memories across multiple topics and valences,
# triggers multi-agent sleep cycles, and captures before/after snapshots
# of beliefs, promotions, and memory stats for manual examination.
#
# Usage:
#   ./scripts/sleep-learning-test.sh [--quick] [--topics TOPIC1,TOPIC2,...]
#
# Options:
#   --quick       Seed only 5 memories per topic (default: 7)
#   --topics      Comma-separated topic list (overrides built-in defaults)
#   --no-wipe     Do not wipe episodic/semantic before starting
#   --cycles N    Number of separate seed+cycle rounds (default: 3)
#   -h, --help    Show this message
#
# Output:
#   All output is tee'd to results/sleep-learning-test-<timestamp>.log
#   Individual snapshots go to results/sleep-learning-test-<timestamp>/
#
# Prerequisites:
#   • aigent daemon running  (aigent start)
#   • aigent binary in PATH  (run install.sh first)
# =============================================================================
set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  BOLD='\033[1m' DIM='\033[2m'
  GREEN='\033[0;32m' YELLOW='\033[0;33m' RED='\033[0;31m'
  CYAN='\033[0;36m' MAGENTA='\033[0;35m' RESET='\033[0m'
else
  BOLD='' DIM='' GREEN='' YELLOW='' RED='' CYAN='' MAGENTA='' RESET=''
fi

h1()   { printf "\n${BOLD}${CYAN}━━━  %s  ━━━${RESET}\n" "$*"; }
h2()   { printf "\n${BOLD}${MAGENTA}── %s ──${RESET}\n" "$*"; }
info() { printf "${GREEN}[+]${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}[!]${RESET} %s\n" "$*"; }
err()  { printf "${RED}[x]${RESET} %s\n" "$*" >&2; }
step() { printf "${BOLD}[→]${RESET} %s\n" "$*"; }
dim()  { printf "${DIM}    %s${RESET}\n" "$*"; }
ts()   { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

# ── Defaults ──────────────────────────────────────────────────────────────────
SEED_COUNT=7        # memories per topic per valence
CYCLES=3            # seed+cycle rounds
WIPE_FIRST=true     # wipe episodic+semantic before starting to get a clean slate
TOPICS=(
  "Rust programming"
  "test-driven development"
  "functional programming"
  "verbose documentation"
  "dynamically typed languages"
)

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)     SEED_COUNT=5; shift ;;
    --no-wipe)   WIPE_FIRST=false; shift ;;
    --cycles)    shift; CYCLES="$1"; shift ;;
    --topics)    shift; IFS=',' read -ra TOPICS <<< "$1"; shift ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,2\}//' | head -20
      exit 0 ;;
    *) err "unknown option: $1"; exit 1 ;;
  esac
done

# ── Output directory setup ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
RESULTS_DIR="${REPO_ROOT}/results/sleep-learning-test-${TIMESTAMP}"
LOG_FILE="${REPO_ROOT}/results/sleep-learning-test-${TIMESTAMP}.log"

mkdir -p "${RESULTS_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"

# Tee all stdout+stderr to the log file from here on
exec > >(tee -a "${LOG_FILE}") 2>&1

# ── Header ────────────────────────────────────────────────────────────────────
h1 "aigent Sleep & Learning Pipeline Test"
printf "  started at   : %s\n" "$(ts)"
printf "  results dir  : %s\n" "${RESULTS_DIR}"
printf "  log file     : %s\n" "${LOG_FILE}"
printf "  seed count   : %d per topic\n" "${SEED_COUNT}"
printf "  cycles       : %d\n" "${CYCLES}"
printf "  wipe first   : %s\n" "${WIPE_FIRST}"
printf "  topics       : %s\n" "${TOPICS[*]}"

# ── Prerequisite check ────────────────────────────────────────────────────────
h2 "Prerequisites"

if ! command -v aigent &>/dev/null; then
  err "aigent is not in PATH — run install.sh first"
  exit 1
fi
info "aigent binary: $(command -v aigent) ($(aigent --version 2>/dev/null || echo unknown))"

step "checking daemon connectivity…"
if ! aigent sleep status &>/dev/null; then
  err "daemon is not running — start it with: aigent start"
  err "(the daemon must be running for sleep cycle commands to work)"
  exit 1
fi
info "daemon is reachable"

# ── Snapshot helper ───────────────────────────────────────────────────────────
snapshot() {
  local label="$1"
  local outdir="${RESULTS_DIR}/${label}"
  mkdir -p "${outdir}"

  step "capturing snapshot: ${label}"

  aigent memory stats                          > "${outdir}/memory_stats.txt"    2>&1 || true
  aigent memory beliefs                        > "${outdir}/beliefs_all.txt"      2>&1 || true
  aigent memory beliefs --tier reflective      > "${outdir}/beliefs_reflective.txt" 2>&1 || true
  aigent memory beliefs --tier semantic        > "${outdir}/beliefs_semantic.txt"   2>&1 || true
  aigent memory beliefs --kind opinion         > "${outdir}/beliefs_opinions.txt"   2>&1 || true
  aigent memory promotions --limit 50          > "${outdir}/promotions.txt"       2>&1 || true
  aigent memory inspect-core --limit 30        > "${outdir}/core.txt"             2>&1 || true
  aigent sleep status                          > "${outdir}/sleep_status.txt"     2>&1 || true

  # Count key metrics for the summary line
  local total beliefs_count opinions_count
  total=$(grep -m1 'total:' "${outdir}/memory_stats.txt" | awk '{print $NF}' || echo '?')
  beliefs_count=$(grep -c '^\[' "${outdir}/beliefs_all.txt" 2>/dev/null || echo '0')
  opinions_count=$(wc -l < "${outdir}/beliefs_opinions.txt" 2>/dev/null || echo '0')

  printf "  %-30s  total=%-6s  beliefs=%-5s  opinions=%s\n" \
    "${label}" "${total}" "${beliefs_count}" "${opinions_count}"
  dim "  saved to: ${outdir}/"
}

# ── Baseline wipe (optional) ──────────────────────────────────────────────────
if [[ "${WIPE_FIRST}" == "true" ]]; then
  h2 "Baseline: wiping episodic + semantic memory for a clean slate"
  warn "this permanently removes episodic and semantic entries from events.jsonl"
  aigent memory wipe --layer episodic  --yes && info "episodic wiped"   || warn "episodic wipe failed (may be empty)"
  aigent memory wipe --layer semantic  --yes && info "semantic wiped"   || warn "semantic wipe failed (may be empty)"
fi

# ── Baseline snapshot ─────────────────────────────────────────────────────────
h2 "Baseline Snapshot (before any seeding)"
snapshot "00_baseline"

# ── Positive preferences (all topics) ─────────────────────────────────────────
h1 "Round 1 — Seed positive preferences across all topics"
printf "  [%s]\n" "$(ts)"

for topic in "${TOPICS[@]}"; do
  h2 "Seeding POSITIVE: \"${topic}\" (${SEED_COUNT} memories)"
  aigent sleep seed "${topic}" --count "${SEED_COUNT}" --valence positive
done

snapshot "01_after_positive_seed"

h2 "Running sleep cycle after positive seeds"
printf "  [%s] starting multi-agent cycle…\n" "$(ts)"
aigent sleep run
printf "  [%s] done\n" "$(ts)"

snapshot "02_after_cycle_1"

# ── Negative/aversion memories (subset of topics) ─────────────────────────────
h1 "Round 2 — Seed contrasting negative preferences"
printf "  [%s]\n" "$(ts)"

# Pick the last 2–3 topics for negative valence to create contrast
NEG_TOPICS=("${TOPICS[@]: -2}")   # last 2 elements

for topic in "${NEG_TOPICS[@]}"; do
  h2 "Seeding NEGATIVE: \"${topic}\" (${SEED_COUNT} memories)"
  aigent sleep seed "${topic}" --count "${SEED_COUNT}" --valence negative
done

# Add one pure neutral observation topic to test non-formation
h2 "Seeding NEUTRAL: \"software architecture\" (${SEED_COUNT} memories)"
aigent sleep seed "software architecture" --count "${SEED_COUNT}" --valence neutral

snapshot "03_after_negative_seed"

h2 "Running sleep cycle after negative seeds"
printf "  [%s] starting multi-agent cycle…\n" "$(ts)"
aigent sleep run
printf "  [%s] done\n" "$(ts)"

snapshot "04_after_cycle_2"

# ── Additional rounds if requested ────────────────────────────────────────────
if [[ "${CYCLES}" -ge 3 ]]; then
  h1 "Round 3 — Reinforce existing preferences (boosts confidence on formed opinions)"
  printf "  [%s]\n" "$(ts)"

  # Re-seed the strong positive topics to test confidence reinforcement
  POS_TOPICS=("${TOPICS[@]:0:2}")   # first 2 elements

  for topic in "${POS_TOPICS[@]}"; do
    h2 "Re-seeding POSITIVE (reinforcement): \"${topic}\" (${SEED_COUNT} memories)"
    aigent sleep seed "${topic}" --count "${SEED_COUNT}" --valence positive
  done

  snapshot "05_after_reinforcement_seed"

  h2 "Running sleep cycle after reinforcement seeds"
  printf "  [%s] starting multi-agent cycle…\n" "$(ts)"
  aigent sleep run
  printf "  [%s] done\n" "$(ts)"

  snapshot "06_after_cycle_3"
fi

# ── Extra cycles if CYCLES > 3 ────────────────────────────────────────────────
for (( i=4; i<=CYCLES; i++ )); do
  h1 "Extra Round ${i} — additional sleep cycle (no new seeds)"
  printf "  [%s]\n" "$(ts)"
  aigent sleep run
  printf "  [%s] done\n" "$(ts)"
  snapshot "$(printf '%02d' $((i * 2)))_after_cycle_${i}"
done

# ── Final summary diff ────────────────────────────────────────────────────────
h1 "Summary: Before vs After"

BASELINE="${RESULTS_DIR}/00_baseline"
FINAL_CYCLE=$(ls -d "${RESULTS_DIR}/"*_after_cycle_* 2>/dev/null | sort | tail -1)

if [[ -n "${FINAL_CYCLE}" ]]; then
  h2 "Memory stats diff (baseline → final)"
  printf "\n%-35s  %-15s  %-15s\n" "metric" "baseline" "final"
  printf "%-35s  %-15s  %-15s\n" "------" "--------" "-----"

  for metric in total core semantic reflective episodic user_profile; do
    b=$(grep -m1 "  ${metric}:" "${BASELINE}/memory_stats.txt" 2>/dev/null | awk '{print $NF}' || echo '?')
    f=$(grep -m1 "  ${metric}:" "${FINAL_CYCLE}/memory_stats.txt" 2>/dev/null | awk '{print $NF}' || echo '?')
    printf "  %-33s  %-15s  %s\n" "${metric}" "${b}" "${f}"
  done

  h2 "New opinions formed (reflective tier)"
  # Show reflective beliefs in final that don't appear in baseline
  if [[ -f "${FINAL_CYCLE}/beliefs_opinions.txt" ]]; then
    if [[ -s "${FINAL_CYCLE}/beliefs_opinions.txt" ]]; then
      cat "${FINAL_CYCLE}/beliefs_opinions.txt"
    else
      warn "no opinion-tier beliefs found — the Identity specialist may need more observations"
      dim  "Check: did each topic receive ≥5 distinct episodic observations?"
      dim  "Tip:   re-run with --cycles 4 or more"
    fi
  fi

  h2 "All reflective beliefs (final)"
  cat "${FINAL_CYCLE}/beliefs_reflective.txt" 2>/dev/null || echo "(none)"

  h2 "Promotions (final)"
  cat "${FINAL_CYCLE}/promotions.txt" 2>/dev/null || echo "(none)"
fi

# ── Completion ────────────────────────────────────────────────────────────────
h1 "Test Complete"
printf "  finished at  : %s\n" "$(ts)"
printf "  log file     : %s\n" "${LOG_FILE}"
printf "  snapshots at : %s\n" "${RESULTS_DIR}/"
printf "\n"
info "Snapshots in order:"
for d in "${RESULTS_DIR}"/*/; do
  printf "  %s\n" "$(basename "${d}")"
done
printf "\n"
info "Suggested next steps:"
dim  "  less ${LOG_FILE}"
dim  "  cat ${FINAL_CYCLE:-${RESULTS_DIR}/missing}/beliefs_opinions.txt"
dim  "  cat ${FINAL_CYCLE:-${RESULTS_DIR}/missing}/beliefs_reflective.txt"
dim  "  diff ${BASELINE}/memory_stats.txt ${FINAL_CYCLE:-${RESULTS_DIR}/missing}/memory_stats.txt"
