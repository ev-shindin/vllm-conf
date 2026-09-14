#!/bin/bash
# Build deep_ep against a pinned NCCL, for deepep_v2 role switching.
#
# Why a rebuild at all: deep_ep's compiled _C tracks the NCCL it was built
# against, and running it against a different runtime gives
# cudaErrorIllegalAddress inside csrc/elastic/buffer.hpp during the first MoE
# forward. The rule is therefore to build against whatever NCCL is installed,
# which is what this does.
#
# It does not force a version. vLLM mirrors NCCL's ncclCommProperties struct --
# the symbol is exported from libnccl but the struct is in no public nccl.h --
# and pynccl_wrapper.py now carries the v2.31.2 layout and clamps the version it
# declares to NCCL_COMM_PROPERTIES_LAYOUT_VERSION. NCCL fills fields gated by
# the version the caller declares rather than by props.size, so the clamp is
# what keeps the runtime from writing past the mirror, and no particular NCCL is
# required on that path any more.
#
# The CUDA major is read, never assumed. vllm/vllm-openai nightlies are CUDA 13
# and carry nvidia-nccl-cu13, so installing an nvidia-nccl-cu12 wheel would put a
# second NCCL of a different major beside the first rather than replacing it,
# leaving the libnccl.so.2 search below to choose between them arbitrarily.
#
# No GPU is needed: nvcc compiles for the arch it is told about.
set -uo pipefail

NCCL_VERSION=${NCCL_VERSION:-}         # empty: build against what is installed
# 9.0 = H100/H200 (sm_90). Deliberately NOT read from TORCH_CUDA_ARCH_LIST,
# which is the variable this exports below: the vllm-openai images already set
# it to every arch they ship for, so reading it back defaulted to nothing and
# the build compiled DeepEP for sm_75 through sm_120. DeepEP's kernels are
# sm_90-only -- elect, mbarrier, cp.async.bulk -- and ptxas rejects the older
# targets outright ("Feature 'elect' requires .target sm_90 or higher").
ARCH=${DEEP_EP_ARCH:-9.0}
OUT=${OUT:-./dist}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)   # absolutise: the build runs from a temp dir
SRC_REF=${SRC_REF:-main}

python3 -c "import importlib.metadata as m; print('### deep_ep before:', m.version('deep_ep'))" 2>/dev/null

# Which NCCL wheel is installed, and for which CUDA major.
NCCL_PKG=$(python3 - <<'PYEOF'
import importlib.metadata as m
for p in ("nvidia-nccl-cu13", "nvidia-nccl-cu12"):
    try:
        m.version(p)
        print(p)
        break
    except Exception:
        pass
PYEOF
)
if [ -z "$NCCL_PKG" ]; then
  echo "### no nvidia-nccl-cu1x wheel installed: nothing to build against"; exit 1
fi

# Only when the caller asks for a specific version, and then of the major that
# is already there -- mixing majors is the failure this guards against.
if [ -n "$NCCL_VERSION" ]; then
  pip install -q --no-cache-dir "${NCCL_PKG}==${NCCL_VERSION}" || {
    echo "### could not install ${NCCL_PKG}==${NCCL_VERSION}"; exit 1; }
fi

# Read the version back rather than echoing one. An earlier revision printed a
# hardcoded line naming a version the build did not use.
NCCL_NOW=$(python3 -c "import importlib.metadata as m; print(m.version('${NCCL_PKG}'))") || exit 1
echo "### building deep_ep against ${NCCL_PKG} ${NCCL_NOW} for sm_${ARCH}"

# nvidia is a NAMESPACE package in the CUDA 13 images, so its __file__ is None
# and os.path.dirname(__file__) raises -- with the error swallowed, leaving an
# empty path that finds no libnccl and aborts below with "missing link library"
# on an image that has one. __path__ is what carries the directories, and a
# namespace package may name more than one.
readarray -t NV_DIRS < <(python3 -c "import nvidia; print('\n'.join(nvidia.__path__))" 2>/dev/null)
if [ ${#NV_DIRS[@]} -eq 0 ]; then
  echo "### could not locate the nvidia package directories"; exit 1
fi
echo "### nvidia package dirs: ${NV_DIRS[*]}"
TK=${CUDA_HOME:-/usr/local/cuda}/targets/x86_64-linux/include

# Some CUDA images ship an incomplete include tree (nvrtc.h and cusparse.h are
# commonly absent while nvcc itself is present). Fill only the GAPS: copying the
# whole wheel include tree shadows the toolkit's own crt/host_runtime.h and
# breaks nvcc's generated stub with a __cudaLaunch arity error.
GAP=$(mktemp -d)
n=0
for d in $(find "${NV_DIRS[@]}" -maxdepth 3 -type d -name include 2>/dev/null); do
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
NCCL_LIB=$(find "${NV_DIRS[@]}" -name "libnccl.so.2" 2>/dev/null | head -1)
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
