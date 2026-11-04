#!/usr/bin/env bash
# Install a pinned Ollama build INSIDE the project venv -- no Homebrew, nothing global.
#
#   binary  -> .venv/ollama/dist/            (wrapper at .venv/bin/ollama)
#   models  -> .venv/ollama/models           (OLLAMA_MODELS)
#   state   -> .venv/ollama/home             (HOME of the ollama process: its key and history)
#
# The wrapper binds the server to 127.0.0.1, so inference never leaves this machine.
# Deleting .venv removes everything. Usage: `make llm-setup` (or bash scripts/setup_ollama.sh).
set -euo pipefail

VERSION="${OLLAMA_VERSION:-v0.35.1}"   # pinned release; override with OLLAMA_VERSION=vX.Y.Z
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
OL="$VENV/ollama"
BASE="https://github.com/ollama/ollama/releases/download/$VERSION"

[ -d "$VENV" ] || { echo "no .venv: run 'make install' first" >&2; exit 1; }
[ "$(uname -s)" = "Darwin" ] || { echo "this script installs the macOS build of Ollama" >&2; exit 1; }

mkdir -p "$OL/models" "$OL/home"
DL="$(mktemp -d "$OL/download.XXXXXX")"   # fresh, empty directory for the untrusted download
trap 'rm -rf "$DL"' EXIT

echo "downloading Ollama $VERSION (macOS CLI build) ..."
curl -fsSL --retry 3 -o "$DL/sha256sum.txt" "$BASE/sha256sum.txt"
echo "  (about 160 MB; this can take a few minutes)"
curl -fsSL --retry 3 -o "$DL/ollama-darwin.tgz" "$BASE/ollama-darwin.tgz"

# verify against the checksum list published with the same release
want="$(awk '$2 ~ /(^|[\/*])ollama-darwin\.tgz$/ {print $1}' "$DL/sha256sum.txt")"
got="$(shasum -a 256 "$DL/ollama-darwin.tgz" | awk '{print $1}')"
if [ -z "$want" ] || [ "$want" != "$got" ]; then
  echo "checksum mismatch for ollama-darwin.tgz (expected '$want', got '$got')" >&2
  exit 1
fi
echo "checksum OK: $got"

rm -rf "$OL/dist" && mkdir -p "$OL/dist"
tar -xzf "$DL/ollama-darwin.tgz" -C "$OL/dist"
BIN="$(find "$OL/dist" -type f -name ollama -perm -u+x | head -n 1)"
[ -n "$BIN" ] || { echo "no ollama binary found in the archive" >&2; exit 1; }

# wrapper: every ollama call (serve, pull, list) uses the venv-local models, state and host
cat > "$VENV/bin/ollama" <<EOF
#!/usr/bin/env bash
# venv-scoped Ollama (written by scripts/setup_ollama.sh). Nothing outside .venv is touched.
export OLLAMA_MODELS="$OL/models"
export OLLAMA_HOST="\${OLLAMA_HOST:-127.0.0.1:11434}"
export HOME="$OL/home"
exec "$BIN" "\$@"
EOF
chmod +x "$VENV/bin/ollama"
echo "$VERSION" > "$OL/VERSION"
echo "installed $VERSION -> $VENV/bin/ollama (models in $OL/models)"
