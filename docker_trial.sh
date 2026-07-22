#!/usr/bin/env bash
# Run one diffusionMTUs translation trial inside the sandbox container.
#
#   ./docker_trial.sh <c_src_dir> [run_project.py args...]
#   e.g. ./docker_trial.sh ../Test-Corpus/Public-Tests/B01_organic/bin2hex_lib/test_case \
#            --config configs/gemma-4-31b.json
#
# Isolation: the repo and the C sources are mounted READ-ONLY; the only
# writable path is $OUT_DIR (default <repo>/out), mounted at /out. The
# container shares the host network namespace (--network=host) so the
# pipeline reaches the local vLLM servers on 127.0.0.1:8000/8001 unchanged.
#
# Env knobs:
#   OUT_DIR       host dir for all state/artifacts (default <repo>/out)
#   MTU_IMAGE     image tag (default diffusionmtus; built if missing)
#   PROJECT_NAME  overrides the derived project name (=> out/_project_<name>)
#   VLLM_BASE_URL / VLLM_API_KEY  passed through when set
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: docker_trial.sh <c_src_dir> [run_project.py args...]" >&2
  exit 64
fi

repo="$(cd "$(dirname "$0")" && pwd)"
c_root="$(cd "$1" && pwd)"; shift
out_dir="$(mkdir -p "${OUT_DIR:-$repo/out}" && cd "${OUT_DIR:-$repo/out}" && pwd)"
image="${MTU_IMAGE:-diffusionmtus}"

# Project name: run_project.py derives it from c_root's grandparent, so the
# mount below recreates <name>/project/<c_root basename> inside the container.
# Derive <name> by climbing past generic directory names, unless overridden.
if [[ -n "${PROJECT_NAME:-}" ]]; then
  name="$PROJECT_NAME"
else
  name_dir="$c_root"
  while [[ "$(basename "$name_dir")" =~ ^(src|test_case|include|c|source)$ ]]; do
    name_dir="$(dirname "$name_dir")"
  done
  name="$(basename "$name_dir")"
fi

if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "[docker_trial] building image $image ..."
  docker build -t "$image" "$repo"
fi

# Local vLLM credentials: mount each api_key_file from config.MODEL_SERVERS
# read-only at its original path so the in-container config resolves as-is.
key_mounts=()
while IFS= read -r key_file; do
  [[ -f "$key_file" ]] && key_mounts+=(-v "$key_file:$key_file:ro")
done < <(python3 - "$repo" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
from config import MODEL_SERVERS
for server in MODEL_SERVERS.values():
    print(server["api_key_file"])
EOF
)

mount_point="/src/$name/project/$(basename "$c_root")"
echo "[docker_trial] project=$name  c_root=$c_root  out=$out_dir"
docker run --rm --init --network=host \
  -v "$repo:/pipeline:ro" \
  -v "$c_root:$mount_point:ro" \
  -v "$out_dir:/out" \
  "${key_mounts[@]}" \
  -e VLLM_BASE_URL="${VLLM_BASE_URL:-}" \
  -e VLLM_API_KEY="${VLLM_API_KEY:-}" \
  "$image" "$mount_point" "$@"
