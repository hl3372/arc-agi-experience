# Sourced (bash) before each vLLM engine starts: pass it as --vllm-env-script. The trainer starts engine g with
# CUDA_VISIBLE_DEVICES=g, i.e. the g-th *physical* GPU. When the run uses a subset of GPUs, set ARC_ENGINE_GPUS to the
# trainer's list (indices or UUIDs, comma-separated, the same as CUDA_VISIBLE_DEVICES) and engine g moves to its g-th
# entry, so it shares a GPU with trainer rank g.
if [ -n "${ARC_ENGINE_GPUS:-}" ]; then
  IFS=, read -ra _ARC_G <<< "$ARC_ENGINE_GPUS"
  export CUDA_VISIBLE_DEVICES=${_ARC_G[$CUDA_VISIBLE_DEVICES]:-}
fi
# FlashInfer JIT-compiles some kernels at engine start. On GPUs newer than the system CUDA toolkit (e.g. sm_103 with a
# CUDA 12.8 nvcc) point CUDA_HOME at a matching toolkit, e.g. the pip "nvidia-cuda-nvcc" of the vLLM environment:
#   export CUDA_HOME=/path/to/vllm-venv/lib/python3.12/site-packages/nvidia/cu13
[ -n "${ARC_VLLM_CUDA_HOME:-}" ] && export CUDA_HOME=$ARC_VLLM_CUDA_HOME
true
