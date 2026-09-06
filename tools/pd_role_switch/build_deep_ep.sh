#!/bin/bash
# Build deep_ep against a pinned NCCL, for deepep_v2 role switching.
#
# Why a rebuild at all: deep_ep's compiled _C tracks the NCCL it was built
# against. The shipped 2.0.0+local in vllm/vllm-openai:v0.28.0 was built against
# NCCL 2.29.7, and running it on a newer runtime gives cudaErrorIllegalAddress
# inside csrc/elastic/buffer.hpp during the first MoE forward.
#
# Why the pin is exact: vLLM hand-mirrors NCCL's ncclCommProperties struct from
# INTERNAL headers -- the symbol is exported from libnccl but the struct is in no
# public nccl.h -- and the mirror is written for the 2.30 layout. ">=2.30.4"
# resolves to the newest wheel, and 2.31.2 corrupts memory in
# ncclCommQueryProperties. The failure is nondeterministic (3 of 8 ranks survived
# it in one run), so a single clean boot does not clear it.
#
# No GPU is needed: nvcc compiles for the arch it is told about.
set -uo pipefail

NCCL_VERSION=${NCCL_VERSION:-2.30.7}
ARCH=${TORCH_CUDA_ARCH_LIST:-9.0}     # 9.0 = H100/H200 (sm_90)
OUT=${OUT:-./dist}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)   # absolutise: the build runs from a temp dir
SRC_REF=${SRC_REF:-main}

echo "### building deep_ep against NCCL ${NCCL_VERSION} for sm_${ARCH}"
python3 -c "import importlib.metadata as m; print('### deep_ep before:', m.version('deep_ep'))" 2>/dev/null

pip install -q --no-cache-dir "nvidia-nccl-cu12==${NCCL_VERSION}" || {
  echo "### could not install nvidia-nccl-cu12==${NCCL_VERSION}"; exit 1; }
# Print the version actually installed. Never report a pin as applied without
# reading it back -- a hardcoded "installed X" line in an earlier version of
# this script claimed the wrong version while the build used another.
python3 -c "import importlib.metadata as m; print('### nccl now:', m.version('nvidia-nccl-cu12'))" || exit 1

NV=$(python3 -c "import os,nvidia; print(os.path.dirname(nvidia.__file__))" 2>/dev/null)
TK=${CUDA_HOME:-/usr/local/cuda}/targets/x86_64-linux/include

# Some CUDA images ship an incomplete include tree (nvrtc.h and cusparse.h are
# commonly absent while nvcc itself is present). Fill only the GAPS: copying the
# whole wheel include tree shadows the toolkit's own crt/host_runtime.h and
# breaks nvcc's generated stub with a __cudaLaunch arity error.
GAP=$(mktemp -d)
n=0
for d in $(find "$NV" -maxdepth 3 -type d -name include 2>/dev/null); do
  ( cd "$d" || exit 0
    for f in $(find . -name "*.h" -o -name "*.hpp" 2>/dev/null); do
      rel=${f#./}
      if [ ! -e "$TK/$rel" ] && [ ! -e "$GAP/$rel" ]; then
        mkdir -p "$GAP/$(dirname "$rel")" 2>/dev/null
        cp "$f" "$GAP/$rel" 2>/dev/null
      fi
    done )
done
n=$(find "$GAP" -name "*.h" 2>/dev/null | wc -l)
echo "### gap-filled ${n} headers the toolkit lacks"
export CPLUS_INCLUDE_PATH="$GAP:${CPLUS_INCLUDE_PATH:-}"
export C_INCLUDE_PATH="$GAP:${C_INCLUDE_PATH:-}"

# libcuda is the DRIVER library and is absent on a build-only machine; the
# toolkit ships a stub for exactly this. libnccl.so.2 comes from the wheel.
STUB=$(find ${CUDA_HOME:-/usr/local/cuda}*/targets/*/lib/stubs -name "libcuda.so*" 2>/dev/null | head -1)
NCCL_LIB=$(find "$NV" -name "libnccl.so.2" 2>/dev/null | head -1)
if [ -z "$STUB" ] || [ -z "$NCCL_LIB" ]; then
  echo "### missing link library: stub=${STUB:-none} nccl=${NCCL_LIB:-none}"; exit 1
fi
export LIBRARY_PATH="$(dirname "$STUB"):$(dirname "$NCCL_LIB"):${LIBRARY_PATH:-}"

TMP=$(mktemp -d)
echo "### fetching DeepEP (${SRC_REF})"
python3 - "$TMP" "$SRC_REF" <<'PYEOF'
import io, sys, tarfile, urllib.request
tmp, ref = sys.argv[1], sys.argv[2]
url = "https://codeload.github.com/deepseek-ai/DeepEP/tar.gz/refs/heads/" + ref
blob = urllib.request.urlopen(url, timeout=180).read()
tarfile.open(fileobj=io.BytesIO(blob)).extractall(tmp)
print("### fetched")
PYEOF
SRCDIR=$(ls -d "$TMP"/DeepEP-* 2>/dev/null | head -1)
[ -z "$SRCDIR" ] && { echo "### fetch failed"; exit 1; }

export TORCH_CUDA_ARCH_LIST="$ARCH"
export MAX_JOBS=${MAX_JOBS:-$(nproc)}
( cd "$SRCDIR" && pip wheel --no-build-isolation --no-deps -w "$OUT" . ) || {
  echo "### build failed"; exit 1; }

echo "### wheel:"
ls -la "$OUT"/deep_ep-*.whl
echo
echo "### install with:"
echo "    pip install --no-deps --force-reinstall $OUT/deep_ep-*.whl"
echo "### then verify from a directory that is NOT the DeepEP source tree:"
echo "    cd /tmp && python3 -c \"import deep_ep; print(hasattr(deep_ep,'ElasticBuffer'))\""
echo "### (importing from inside the source tree resolves to the uncompiled"
echo "###  sources and prints a traceback for a build that actually succeeded)"
