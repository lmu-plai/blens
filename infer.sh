#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
XFL="/home/tristan/Documents/App/xfl/xfl-r/XFL"
IDAB="/home/tristan/Documents/idapro-7.5/idat64"
BLENS="/home/tristan/Documents/App/blens"
BLENS_DATA="/home/tristan/Documents/App/data"

# ---------- Input ----------
TARGET="${XFL}/data/binaries/"

# ---------- Other variables ----------
ENV_XFL="XFL4"
ENV_BLENS="BLens"
NLP_DATA="${BLENS_DATA}/nlpDataTest"

# ---------- Ensure conda is available ----------
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH. Load/initialize Conda and retry." >&2
  exit 1
fi

# Prefer streaming output if supported by your conda version
if conda run --help 2>/dev/null | grep -q -- '--no-capture-output'; then
  CRUN=(conda run --no-capture-output)
else
  CRUN=(conda run)
fi

# ---------- DEXTER stage ----------
pushd "$XFL" >/dev/null
"${CRUN[@]}" -n "$ENV_XFL" python3 add_binaries.py -p "${TARGET}"
"${CRUN[@]}" -n "$ENV_XFL" python3 dexter.py -d origin -exportEpoch 50 -inferenceMode -blensNlpMode -batchSize 16
popd >/dev/null

# Move results to BLens data locations
mv "${XFL}/res/origin/nlpData" "${NLP_DATA}"
mv "${XFL}/res/origin/embeddings" "${BLENS_DATA}/embedding/dexter_test"

# ---------- CLAP stage ----------
pushd "${BLENS}/preprocessing/clap" >/dev/null
"${CRUN[@]}" -n "$ENV_BLENS" python3 main.py \
  --batch-size 16 \
  --nlpData="${NLP_DATA}" \
  --idaB="${IDAB}" \
  --output="${BLENS_DATA}/embedding/clap_test"
popd >/dev/null

# ---------- PalmTree stage ----------
pushd "${BLENS}/preprocessing/palmtree" >/dev/null
"${CRUN[@]}" -n "$ENV_BLENS" python3 main.py "${NLP_DATA}" "${BLENS_DATA}/palmtree/" "${BLENS_DATA}/embedding/palmtree_test"
popd >/dev/null

# ---------- BLens stage ----------
pushd "${BLENS}" >/dev/null
"${CRUN[@]}" -n "$ENV_BLENS" python3 Infer.py --batch-size 16 -d="V11-BINARIES-NO+UNK-DECODER+MULTI-LONG++" --cross-binary  -data-dir="${BLENS_DATA}"
popd >/dev/null

echo "Pipeline finished successfully."
