#!/usr/bin/env bash
# Builds the isolated conda env this pilot's slime/Megatron/SGLang stack runs in, on a bare-metal
# single-B200 node (no docker available here -- see docs/pilot_setup.md §2's docker-vs-conda
# fallback). Adapted from vendor/ICRL/build_conda.sh (which assumes running as root inside the
# slimerl/slime docker build context) for this node:
#   - env is created with plain `conda`, not micromamba-bootstrapped-from-curl
#   - reuses the system CUDA 12.8 toolkit already at /usr/local/cuda-12.8 (nvcc present) instead of
#     installing a second CUDA toolchain via conda -- driver reports max CUDA 12.8 (nvidia-smi), so
#     torch/extensions are pinned to cu128 here, not build_conda.sh's cu129
#   - BASE_DIR defaults to a scratch dir OUTSIDE both this repo and the vendored ICRL tree (source
#     clones of sglang/Megatron-LM are multi-GB and must never be committed)
#   - vendor/ICRL/slime already exists in-tree (ICRL's own fork of slime), so this installs THAT
#     copy editable (`pip install -e` from vendor/ICRL), matching build_conda.sh's "else" branch --
#     it does not clone a second, separate THUDM/slime checkout
#
# This ONLY touches the `icrl-pilot` conda env (created separately: `conda create -n icrl-pilot
# python=3.12 pip -y`). It never runs `uv`/`pip` against /home/dg793/steerable-text-to-lora's own
# .venv, and never installs anything into any other conda env on this node.
#
#   conda activate icrl-pilot && bash setup_icrl_pilot_env.sh
set -euxo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "icrl-pilot" ]]; then
    echo "error: activate the icrl-pilot conda env first (conda activate icrl-pilot)" >&2
    exit 1
fi

ICRL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/vendor/ICRL" &>/dev/null && pwd)"
BASE_DIR="${BASE_DIR:-$HOME/icrl_pilot_build}"
mkdir -p "$BASE_DIR"

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

# build_conda.sh gets NCCL dev headers from a `micromamba install nccl` (conda package, headers
# land in $CONDA_PREFIX/include which the conda toolchain's compiler wrappers auto-search). We
# don't do that conda install here (using the system CUDA 12.8 toolchain instead), so headers must
# come from somewhere else -- torch's `nvidia-nccl-cu12` pip dependency (pulled in transitively by
# `torch` above) ships nccl.h too, just under site-packages/nvidia/nccl/include rather than a
# standard search path. Without this, transformer_engine_torch's build fails with
# "fatal error: nccl.h: No such file or directory" (hit on the first attempt at this script).
NCCL_INCLUDE_DIR="$(python3 -c 'import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], "include"))')"
export CPATH="${NCCL_INCLUDE_DIR}:${CPATH:-}"

# Idempotent patch apply -- BASE_DIR's source checkouts persist across re-runs of this script
# (e.g. resuming on a different node after the first attempt got interrupted), so a plain
# `git apply` fails with "patch does not apply" the second time round on an already-patched tree.
# `git apply --reverse --check` succeeds silently iff the patch is already applied.
apply_patch_idempotent() {
    local patch_file="$1"
    if git apply --reverse --check "$patch_file" 2>/dev/null; then
        echo "  (patch already applied, skipping) $patch_file"
    else
        git apply "$patch_file"
    fi
}

echo "=== torch (cu128, matching this node's driver-reported max CUDA 12.8)"
pip install cuda-python==12.8.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

echo "=== sglang (source, pinned commit + ICRL's v0.5.7 patch)"
cd "$BASE_DIR"
if [[ ! -d sglang ]]; then
    git clone https://github.com/sgl-project/sglang.git
fi
cd sglang
git checkout "${SGLANG_COMMIT}"
apply_patch_idempotent "$ICRL_DIR/docker/patch/v0.5.7/sglang.patch"
pip install -e "python[all]"

pip install cmake ninja

echo "=== flash-attn (newest Megatron supports)"
MAX_JOBS="${MAX_JOBS:-8}" pip install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.0
NVCC_APPEND_FLAGS="--threads 4" \
  pip install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

echo "=== Megatron-LM (source, pinned commit + ICRL's v0.5.7 patch)"
cd "$BASE_DIR"
if [[ ! -d Megatron-LM ]]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd Megatron-LM
git checkout "${MEGATRON_COMMIT}"
apply_patch_idempotent "$ICRL_DIR/docker/patch/v0.5.7/megatron.patch"
pip install -e .

echo "=== slime (ICRL's in-tree fork, editable) + icrl's own requirements"
cd "$ICRL_DIR"
pip install -e .
pip install -r requirements.txt

# icrl/envs/math_env.py imports math_verify directly but it's absent from requirements.txt (likely
# baked into the docker base image ICRL's own build_conda.sh assumes) -- without this,
# icrl.hydra_runner fails immediately with "ModuleNotFoundError: No module named 'math_verify'".
pip install math_verify

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"

# The numpy<2 downgrade above breaks scipy>=1.14 (built against numpy>=2.0, uses np.long which
# numpy<2 doesn't have) -- and transformers imports scipy transitively at module load (via its
# object-detection loss code), so without this pin `from transformers import PreTrainedModel`
# (needed by mbridge, used by tools/convert_hf_to_torch_dist.py) fails with
# "AttributeError: module 'numpy' has no attribute 'long'" (hit converting the Qwen3-4B checkpoint).
pip install "scipy<1.14"

echo "=== done -- icrl-pilot env ready"
python3 -c "import torch, sglang, megatron, slime; print('torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())"
