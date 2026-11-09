"""Shared fixtures of the RL tests: a byte-level stub tokenizer, a tiny random Qwen3.5 checkpoint (torch +
transformers), the flag set of the GRPO contracts, and a rollout_step harness with a fake sampler."""
import json
import os
from types import SimpleNamespace as NS

from arcsolver.prompt import arc_render as ar
from arcsolver.rl import rewards as rr


# --- a byte-level stub tokenizer --------------------------------------------------------------------------------------

class _Added:
    def __init__(self, content):
        self.content = content


class StubTok:
    """Byte-level BPE stand-in: id 1000+b = byte b; a few multi-char words; "</think>" (added, not special) and
    "<|im_end|>" (special). Non-ASCII chars become several single-byte tokens (split UTF-8 sequences)."""
    THINK, END = 5, 6
    WORDS = ["\n\n", "pair", " was", "Check", " cell", "row"]

    def __init__(self):
        self.b2u = rr._bytes_to_unicode()
        self.added_tokens_decoder = {self.THINK: _Added("</think>"), self.END: _Added("<|im_end|>")}
        self.all_special_ids = [self.END]

    def convert_ids_to_tokens(self, i):
        if 2000 <= i < 2000 + len(self.WORDS):
            return "".join(self.b2u[b] for b in self.WORDS[i - 2000].encode())
        return self.b2u[i - 1000]

    def encode(self, text):
        ids, pos = [], 0
        while pos < len(text):
            if text.startswith("</think>", pos):
                ids.append(self.THINK)
                pos += 8
                continue
            if text.startswith("<|im_end|>", pos):
                ids.append(self.END)
                pos += 10
                continue
            for k, w in enumerate(self.WORDS):
                if text.startswith(w, pos):
                    ids.append(2000 + k)
                    pos += len(w)
                    break
            else:
                ids.extend(1000 + b for b in text[pos].encode())
                pos += 1
        return ids


def token_text(stub, text):
    return rr.TokenAligner(stub)(stub.encode(text))


# --- a tiny Qwen3.5 checkpoint ------------------------------------------------------------------------------------------

TINY_QWEN35 = {
    "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "language_model_only": False,
    "image_token_id": 1000, "video_token_id": 1000, "vision_start_token_id": 1000, "vision_end_token_id": 1000,
    "tie_word_embeddings": False,
    "text_config": {
        "model_type": "qwen3_5_text", "attention_bias": False, "attention_dropout": 0.0, "attn_output_gate": True,
        "bos_token_id": 1, "eos_token_id": 1, "pad_token_id": None, "dtype": "bfloat16", "full_attention_interval": 4,
        "head_dim": 64, "hidden_act": "silu", "hidden_size": 256, "initializer_range": 0.02, "intermediate_size": 512,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"], "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 64, "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_value_head_dim": 64,
        "mamba_ssm_dtype": "float32", "max_position_embeddings": 4096, "mtp_num_hidden_layers": 0,
        "mtp_use_dedicated_embeddings": False, "num_attention_heads": 4, "num_hidden_layers": 4,
        "num_key_value_heads": 2, "output_gate_type": "swish", "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-6,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000, "rope_type": "default"},
        "tie_word_embeddings": False, "use_cache": True, "vocab_size": 1024},
    "vision_config": {
        "model_type": "qwen3_5", "deepstack_visual_indexes": [], "depth": 1, "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 64, "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 128, "num_heads": 2,
        "num_position_embeddings": 2304, "out_hidden_size": 256, "patch_size": 16, "spatial_merge_size": 2,
        "temporal_patch_size": 2},
}


def tiny_qwen35(d):
    """Write a randomly initialised 4-layer Qwen3.5 (3 GatedDeltaNet + 1 attention layer, vocab 1024) to `d`."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(TINY_QWEN35, f)
    torch.manual_seed(0)
    AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(d), dtype=torch.bfloat16).save_pretrained(d)


# --- the GRPO contract flags ---------------------------------------------------------------------------------------------

CONTRACT_FLAGS = """--prompts-per-step 250 --group-size 6 --train-split-tests
--adv-norm std --adv-eps 1e-4 --adv-clip 0 --loss-agg token
--pg-loss ppo --sc-topk 0 --clip-eps 0.2 --clip-eps-high 0.2
--drop-zero-var outcome --minibatch-max 1 --temperature 1
--w-outcome 1 --outcome-require-stop 1 --w-format 0
--beta-loop 0 --w-loop-seq 0 --w-len 0
--len-abs-alpha 0 --len-correct-alpha 0 --len-share-lambda 0
--len-allcorrect-beta 0 --len-pair-lambda 0
--mask-truncated 0 --trunc-neg-scale 1 --train-budget-force 0
--dyn-target-groups 0 --prefix-groups 0
--train-force-inline 0 --prefix-extra-prompts 0 --per-request-seeds""".split()
CONTRACT_BASE = ["--out", "/tmp/rl_contract_test", "--student", "/tmp/none", "--objective-contract", "grpo_token_v1"]


# --- rollout_step with a fake sampler ------------------------------------------------------------------------------------

HARNESS_OUT = ["--out", "/tmp/rl_harness_test", "--student", "/tmp/none"]
G = 4
OUTCOME = {0: (2, 0), 1: (4, 0), 2: (0, 4), 3: (0, 1), 4: (1, 3), 5: (0, 2)}    # item -> (n correct, n truncated)
RIGHT, WRONG = "Rule.\n</think>\n\n1", "Rule.\n</think>\n\n0"


def simple_task():
    return {"train": [{"input": [[0]], "output": [[1]]}], "test": [{"input": [[0]], "output": [[1]]}]}


def rollout_harness(monkeypatch, extra=()):
    """(args, pool, step): step(order, n) runs train.rollout_step for step n with a fake sampler (sample j of item k is
    truncated if j < n_trunc and right if j >= G - n_correct, per OUTCOME), then the frontier update as main() does."""
    from arcsolver.rl import train as rd
    a = rd.parse_args(HARNESS_OUT + ["--group-size", str(G), "--prompts-per-step", "3", "--augment", "0",
                                     "--seed", "11"] + list(extra))
    ar.configure_arm(a.arm)
    t = simple_task()
    pool = [NS(task_id=f"t{k}", path=f"t{k}.json", student_ids=[100 + k], seen=False, rate=None, source="bank")
            for k in OUTCOME]
    monkeypatch.setattr(rd.RT, "augment_group", lambda args, task, it, *x: (task, it.student_ids, NS(), 0))

    def call(method, **kw):
        assert method == "generate" and kw["n_per_prompt"] == [G] * len(kw["prompts"])
        return dict(outputs=[[dict(token_ids=[p[0], j], logprobs=[-.5, -.5], text="",
                                   finish="length" if j < OUTCOME[p[0] - 100][1] else "stop") for j in range(G)]
                             for p in kw["prompts"]], gen_tokens=0, seconds=1.)

    def align(ids):
        k, j = ids[0] - 100, ids[1]
        s = RIGHT if j >= G - OUTCOME[k][0] else WRONG
        return rr.TokenText(s, list(range(len(s))), list(range(1, len(s) + 1)))

    cfg = rr.RewardConfig(beta_loop=0, outcome_all_tests=True)

    def step(order, n):
        out = rd.rollout_step(a, n, order, pool, lambda p: t, None, NS(call=call), cfg, align)
        assert len(out) == 7                                                 # the return signature
        if a.pool_order == "frontier":
            order.update(rd.frontier_results(out[0]))
        return out
    return a, pool, step


def fp8_checkpoint(src, dst):
    """bf16 HF dir -> FP8 sampler dir with fp8_quant (block-128 e4m3 weights + scales, quantization_config in config.json),
    the format SyncWriterFP8 writes and --sampler-model expects with --sampler-fp8 1."""
    import shutil
    from safetensors import safe_open
    from safetensors.torch import save_file
    from arcsolver.rl import fp8_quant as fq

    def entries(name, w):
        out = fq.quantize_tensor_entries(name, w) if hasattr(fq, "quantize_tensor_entries") else None
        if out is not None:
            return out
        if fq.should_quantize(name, tuple(w.shape)):
            q, s = fq.quantize_weight(w)
            return {name: q, fq.scale_name(name): s}
        return {name: w}

    os.makedirs(dst, exist_ok=True)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if not (fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or os.path.isdir(p)):
            shutil.copy2(p, os.path.join(dst, fn))
    idx_path = os.path.join(src, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx_path))["weight_map"].values())) if os.path.exists(idx_path) \
        else ["model.safetensors"]
    weight_map, total = {}, 0
    for fn in files:
        out = {}
        with safe_open(os.path.join(src, fn), "pt") as f:
            for k in f.keys():
                for kk, v in entries(k, f.get_tensor(k)).items():
                    out[kk] = v.contiguous()
                    weight_map[kk] = fn
                    total += v.numel() * v.element_size()
        save_file(out, os.path.join(dst, fn), metadata={"format": "pt"})
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)
    cfg_path = os.path.join(dst, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["quantization_config"] = fq.fp8_quantization_config()
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def safetensors_dir(d):
    """{tensor name: tensor} of every *.safetensors file in d."""
    from safetensors import safe_open
    out = {}
    for fn in os.listdir(d):
        if fn.endswith(".safetensors"):
            with safe_open(os.path.join(d, fn), "pt") as f:
                for k in f.keys():
                    out[k] = f.get_tensor(k)
    return out


def raw_bytes(t):
    import torch
    return t.view(torch.uint8) if t.element_size() == 1 else t
