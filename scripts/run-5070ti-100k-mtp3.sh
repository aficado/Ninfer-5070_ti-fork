#!/usr/bin/env bash
#
# Qwen3.8-27B (gsq3) on the RTX 5070 Ti (16 GB, sm_120a) with MTP3 speculative decoding
# at the documented 100K-token profile.
#
# This is the native (non-Docker) launcher for this fork: the published 4080 launchers are
# sm_89 only and do not run on this card. It runs the locally built build/apps/ninfer-serve
# binary against the compiled gsq3 artifact, exactly as documented in the repo README:
#
#   --host 127.0.0.1 --port 8080
#   --max-context 102400 --kv-capacity 102400 --kv-dtype rk4v4-e8
#   --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 2688 --host-kv-mib 4096
#   --spec mtp --draft-tokens 3 --lm-head-draft --vision --preserve-thinking
#
# Measured on this card (102,400 tokens, MTP3): weights 11.2 GiB, ~1.48 GiB free after
# startup; MTP acceptance 49-65% on real 512-output prompts, ~112-132 tok/s decode.
#
# Overrides: NINFER_SERVER (binary path), NINFER_MODEL_DIR (model location),
# NINFER_PORT (default 8080), NINFER_BIND (default 127.0.0.1),
# NINFER_API_KEY (optional bearer token), NINFER_DEFAULT_MAX_TOKENS,
# NINFER_DEFAULT_THINKING_BUDGET.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server="${NINFER_SERVER:-$root/build/apps/ninfer-serve}"
model="${1:-${NINFER_MODEL_DIR:-$root/scripts/models}/qwen3_8_27b_gsq3.ninfer}"
host_port="${NINFER_PORT:-8080}"
bind_address="${NINFER_BIND:-127.0.0.1}"

if [[ ! -x "$server" ]]; then
    printf 'Missing executable: %s\n' "$server" >&2
    printf 'Build it first with scripts/build-5070ti.sh /path/to/cuda-13.x\n' >&2
    exit 1
fi
if [[ ! -f "$model" ]]; then
    printf 'Missing model: %s\n' "$model" >&2
    printf 'Download it first with scripts/download-qwen38-gsq3.sh\n' >&2
    exit 1
fi

api_args=()
if [[ -n "${NINFER_API_KEY:-}" ]]; then
    api_args=(--api-key "$NINFER_API_KEY")
fi

extra_args=()
if [[ -n "${NINFER_DEFAULT_MAX_TOKENS:-}" ]]; then
    extra_args+=(--default-max-tokens "$NINFER_DEFAULT_MAX_TOKENS")
fi
if [[ -n "${NINFER_DEFAULT_THINKING_BUDGET:-}" ]]; then
    extra_args+=(--default-thinking-budget "$NINFER_DEFAULT_THINKING_BUDGET")
fi

printf 'Serving Qwen3.8-27B gsq3 (MTP3) at %s tokens on http://%s:%s/v1\n' \
    "102400" "$bind_address" "$host_port"

# If the system CUDA runtime is older than 13, put the CUDA 13 libraries on LD_LIBRARY_PATH.
if [[ -n "${NINFER_CUDA_LIBS:-}" ]]; then
    export LD_LIBRARY_PATH="$NINFER_CUDA_LIBS:$LD_LIBRARY_PATH"
fi

exec "$server" "$model" \
    --host "$bind_address" --port "$host_port" \
    --max-context 102400 --kv-capacity 102400 --kv-dtype rk4v4-e8 \
    --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 2688 --host-kv-mib 4096 \
    --spec mtp --draft-tokens 3 --lm-head-draft --vision --preserve-thinking \
    "${extra_args[@]+"${extra_args[@]}"}" \
    "${api_args[@]+"${api_args[@]}"}"
