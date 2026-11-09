"""The GRPO additions: the frozen KL reference (fsdp.load(frozen=True)), accepted alternative answers ("alt_outputs":
scoring, augmentation, never rendered), the grpo_dapo_* contracts, --pool-drop-solved(-seed), the soft overlong
penalty, the LR ramp, node-affine minibatch plans and the trainer's own old log-probs (--ppo-old-logp trainer)."""
import copy
import gzip
import json
import os
import socket
import tempfile
import types
from types import SimpleNamespace as NS

import pytest

from arcsolver.prompt import arc_render as ar
from arcsolver.rl import augment as ra
from arcsolver.rl import config as RT
from arcsolver.rl import contract as C
from arcsolver.rl import core as oc
from arcsolver.rl import pool as rp
from arcsolver.rl import rewards as rr
from arcsolver.rl import train as D
from arcsolver.rl import workers

from .helpers import CONTRACT_BASE as BASE
from .helpers import CONTRACT_FLAGS, rollout_harness, tiny_qwen35


def _needs_torch_and_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    return torch


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- the frozen KL reference (torch + transformers) ---------------------------------------------------------------

def test_frozen_reference_equals_policy_at_start():
    """The bf16 frozen reference and the fp32-master trainer compute in the same bf16: identical log-probs at step 0
    (so the KL is exactly 0 before the first update), the reference has no trainable parameters, and it moves
    to CPU and back (timeshare)."""
    torch = _needs_torch_and_transformers()
    import torch.distributed as dist
    torch.backends.mkldnn.enabled = False
    d = tempfile.mkdtemp(prefix="klref_")
    tiny_qwen35(d)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(_free_port()), RANK="0", WORLD_SIZE="1")
    dist.init_process_group("gloo")
    try:
        from arcsolver.rl import fsdp as F
        from arcsolver.rl import loss as rl
        dev = torch.device("cpu")
        model = F.load(d, dev, chunk=8, keep_gathered=False, freeze_embed=True)
        ref = F.load(d, dev, chunk=8, keep_gathered=True, frozen=True)
        assert not any(p.requires_grad for p in ref.parameters())
        assert all(p.dtype == torch.bfloat16 for p in ref.parameters())
        ids = torch.randint(0, 1000, (1, 20), generator=torch.Generator().manual_seed(3))
        F.gather_frozen(ref, dev)                    # the step's one collective; forwards then stay local
        with torch.no_grad():
            lr_ = ref(ids, 8).float()
            assert torch.equal(ref(ids, 8).float(), lr_)
        F.reshard_frozen(ref)
        lp = model(ids, 8)
        assert torch.equal(lr_, lp.detach().float())
        loss, st = rl.kl_k3(lp, lr_, torch.ones_like(lp), float(lp.numel()), 0.001)
        assert float(loss) == 0.0
        ref.to("cpu")
        with torch.no_grad():
            assert torch.equal(ref(ids, 8).float(), lr_)
    finally:
        dist.destroy_process_group()


def _unequal_ref_forwards(rank, world, port, d, keep, out):
    import datetime
    import torch
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    torch.backends.mkldnn.enabled = False
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=60))
    try:
        from arcsolver.rl import fsdp as F
        dev = torch.device("cpu")
        ref = F.load(d, dev, chunk=8, keep_gathered=keep, frozen=True)
        F.gather_frozen(ref, dev)
        ids = torch.randint(0, 1000, (1, 16), generator=torch.Generator().manual_seed(rank))
        with torch.no_grad():
            lps = [ref(ids, 6).float() for _ in range(3 if rank == 0 else 1)]   # --balance lpt: unequal counts
        F.reshard_frozen(ref)
        dist.barrier()
        with open(out + f".{rank}", "w") as f:
            f.write(f"{len(lps)} {float(lps[0].sum()):.6f}")
    finally:
        dist.destroy_process_group()


def _run_two_ranks(target, *args):
    import torch.multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=target, args=(r, 2, *args)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=240)
    alive = [p.is_alive() for p in procs]
    for p in procs:
        if p.is_alive():
            p.kill()
    return alive, [p.exitcode for p in procs]


def test_reference_forwards_need_no_collectives_across_unequal_ranks():
    """--balance lpt gives ranks different numbers of rollouts: the reference's per-rollout forwards must not issue
    collectives (one gather per minibatch, then local forwards), or the ranks deadlock."""
    _needs_torch_and_transformers()
    d = tempfile.mkdtemp(prefix="klref2_")
    tiny_qwen35(d)
    out = os.path.join(d, "res")
    alive, codes = _run_two_ranks(_unequal_ref_forwards, _free_port(), d, True, out)
    assert not any(alive), "ranks hung with unequal reference forwards"
    assert all(c == 0 for c in codes), codes
    assert open(out + ".0").read().startswith("3 ") and open(out + ".1").read().startswith("1 ")


def _trainer_old_logp(rank, world, port, d, keep, out):
    import datetime
    import torch
    import torch.distributed as dist
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    torch.backends.mkldnn.enabled = False
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=60))
    try:
        from arcsolver.rl import fsdp as F
        dev = torch.device("cpu")
        model = F.load(d, dev, chunk=8, keep_gathered=keep)
        g = torch.Generator().manual_seed(rank)
        items = [{"prompt_ids": torch.randint(0, 1000, (6,), generator=g).tolist(),
                  "completion": torch.randint(0, 1000, (10,), generator=g).tolist()} for _ in range(3 if rank == 0 else 1)]
        if not keep:                                                   # every forward a collective: pad to equal counts
            items = items + [None] * (3 - len(items))
        old = F.trainer_logprobs(model, dev, items)
        assert all(not lp.requires_grad for lp in old if lp is not None)
        # the training forward right after (same weights) gives the same log-probs, with a gradient
        F.begin_step(model)
        it = items[0]
        lp = model(torch.tensor([it["prompt_ids"] + it["completion"]]), 6)
        lp.sum().backward()
        dist.barrier()
        with open(out + f".{rank}", "w") as f:
            f.write(f"{sum(x is not None for x in old)} {float((lp.detach().float() - old[0]).abs().max()):.3g}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("keep", [True, False])
def test_trainer_old_logprobs_unequal_ranks(keep):
    _needs_torch_and_transformers()
    d = tempfile.mkdtemp(prefix="oldlp_")
    tiny_qwen35(d)
    out = os.path.join(d, "res")
    alive, codes = _run_two_ranks(_trainer_old_logp, _free_port(), d, keep, out)
    assert not any(alive), "ranks hung in the old log-prob forwards"
    assert all(c == 0 for c in codes), codes
    for r, n in ((0, 3), (1, 1)):
        k, diff = open(out + f".{r}").read().split()
        assert int(k) == n and float(diff) < 1e-4


# ---- alternative answers ------------------------------------------------------------------------------------------

def _task():
    return {"train": [{"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]}],
            "test": [{"input": [[2, 0], [0, 0]], "output": [[0, 2], [0, 0]],
                      "alt_outputs": [[[2, 2], [0, 0]], [[0, 0, 0], [0, 2, 0]]]}]}


def _answer(grid):
    return "</think>\n\n" + "\n".join("".join(map(str, r)) for r in grid)


def test_alt_outputs_count_as_correct():
    ar.configure_arm("r2k_zb_max")
    t = _task()
    for g, ok in (([[0, 2], [0, 0]], True), ([[2, 2], [0, 0]], True), ([[0, 0, 0], [0, 2, 0]], True),
                  ([[2, 0], [0, 2]], False)):
        e = rr.answer_eval(t, _answer(g), "stop")
        assert e["em_all"] is ok, (g, e)
    e = rr.answer_eval(t, _answer([[1, 1, 1], [1, 1, 1]]), "stop")    # the shape check accepts an alternative's shape
    assert e["shape_ok"] and not e["em_all"]
    plain = copy.deepcopy(t)
    del plain["test"][0]["alt_outputs"]
    assert not rr.answer_eval(plain, _answer([[2, 2], [0, 0]]), "stop")["em_all"]


def test_alt_outputs_follow_the_augmentation_and_are_never_rendered():
    t = _task()
    aug = ra.Aug("rot90", tuple([0, 3, 5, 1, 2, 4, 6, 7, 8, 9]))
    at = aug.task(t)
    assert at["test"][0]["alt_outputs"] == [aug.grid(g) for g in t["test"][0]["alt_outputs"]]
    assert at["test"][0]["output"] == aug.grid(t["test"][0]["output"])
    assert t["test"][0]["alt_outputs"][0] == [[2, 2], [0, 0]]          # the original is untouched
    plain = copy.deepcopy(t)
    del plain["test"][0]["alt_outputs"]
    assert oc.student_messages(t) == oc.student_messages(plain)
    assert oc.task_fingerprint(t) == oc.task_fingerprint(plain)


# ---- contracts ----------------------------------------------------------------------------------------------------

def _base(contract, *extra):
    return [x if x != "grpo_token_v1" else contract for x in BASE] + list(extra)


def test_grpo_dapo_kl_contract():
    flags = list(CONTRACT_FLAGS)
    flags[flags.index("--train-budget-force") + 1] = "2048"
    flags[flags.index("--prompts-per-step") + 1] = "272"
    flags += ["--pool-drop-solved", "1"]
    base = _base("grpo_dapo_kl_v1", "--lr", "2e-6", "--warmup-steps", "10", "--eval-temperature", "1.0",
                 "--eval-top-p", "1.0", "--eval-top-k", "0")
    a = D.parse_args(base + flags + ["--kl-coef", "0.001", "--forced-credit", "0.5"])
    assert C.check_contract(a, "grpo_dapo_kl_v1") == []
    assert a.ref_model == a.student and a.kl_coef == 0.001 and a.pool_order == "weighted"
    with pytest.raises(SystemExit):                                     # forced credit 0.75 is not this contract
        D.parse_args(base + flags + ["--kl-coef", "0.001", "--forced-credit", "0.75"])
    with pytest.raises(SystemExit):                                     # no KL
        D.parse_args(base + flags + ["--forced-credit", "0.5"])
    with pytest.raises(SystemExit):                                     # a tiered order
        D.parse_args(base + flags + ["--kl-coef", "0.001", "--forced-credit", "0.5", "--pool-order", "frontier"])
    with pytest.raises(SystemExit):                                     # the reference must be the start checkpoint
        D.parse_args(base + flags + ["--kl-coef", "0.001", "--forced-credit", "0.5", "--ref-model", "/tmp/other"])
    with pytest.raises(SystemExit):                                     # the contract pins the LR
        D.parse_args(base + flags + ["--kl-coef", "0.001", "--forced-credit", "0.5", "--lr", "3e-6"])
    snap = C.objective_snapshot(a)
    assert snap["values"]["kl_coef"] == 0.001 and snap["values"]["ref_model"] == "/tmp/none"


def test_grpo_dapo_kl_v2_drops_forced_answers():
    flags = list(CONTRACT_FLAGS)
    flags[flags.index("--prompts-per-step") + 1] = "272"
    flags += ["--pool-drop-solved", "1"]
    base = _base("grpo_dapo_kl_v2", "--lr", "2e-6", "--warmup-steps", "10")
    a = D.parse_args(base + flags + ["--kl-coef", "0.001"])            # --train-budget-force 0
    assert C.check_contract(a, "grpo_dapo_kl_v2") == [] and a.train_budget_force == 0
    flags2 = list(flags)
    flags2[flags2.index("--train-budget-force") + 1] = "2048"
    with pytest.raises(SystemExit):
        D.parse_args(base + flags2 + ["--kl-coef", "0.001"])


def _sc_flags():
    flags = list(CONTRACT_FLAGS)
    flags[flags.index("--prompts-per-step") + 1] = "272"
    flags[flags.index("--pg-loss") + 1] = "tis_sc"
    flags[flags.index("--sc-topk") + 1] = "32"
    return flags + ["--pool-drop-solved", "1", "--tis-cap", "2.0", "--kl-coef", "0.001"]


def test_grpo_dapo_kl_sc_contract():
    """Contract grpo_dapo_kl_sc_v1 (tis_sc, top-32, TIS cap 2, KL kept): the 9B recipe's first phase."""
    base = _base("grpo_dapo_kl_sc_v1", "--lr", "2e-6", "--warmup-steps", "10")
    a = D.parse_args(base + _sc_flags())
    assert C.check_contract(a, "grpo_dapo_kl_sc_v1") == [] and a.pg_loss == "tis_sc" and a.kl_coef == 0.001
    bad = _sc_flags()
    bad[bad.index("--sc-topk") + 1] = "0"
    with pytest.raises(SystemExit):
        D.parse_args(base + bad)


PREFIX = ["--prefix-every", "2", "--prefix-max-zero", "11", "--prefix-zero-max-frac", "0.5", "--prefix-early-frac", "0.6",
          "--prefix-early-prob", "0.8", "--prefix-end-margin", "10000"]


def test_grpo_dapo_kl_sc_prefix_contract():
    """The sc_v1 objective plus prefix groups every two steps (68 of 272): the 9B recipe's second phase."""
    base = _base("grpo_dapo_kl_sc_prefix_v1", "--lr", "2e-6", "--warmup-steps", "10")
    flags = _sc_flags()
    flags[flags.index("--prefix-groups") + 1] = "68"
    a = D.parse_args(base + flags + PREFIX)
    assert C.check_contract(a, "grpo_dapo_kl_sc_prefix_v1") == []
    assert (a.prefix_groups, a.prefix_every, a.prefix_max_zero) == (68, 2, 11)
    assert any("prefix_groups" in x for x in C.check_contract(a, "grpo_dapo_kl_sc_v1"))
    with pytest.raises(SystemExit):                                  # every step is not this contract
        D.parse_args(base + flags + [x if x != "2" else "1" for x in PREFIX])


def test_grpo_dapo_kl_sc_prefix_v2_contract():
    base = _base("grpo_dapo_kl_sc_prefix_v2", "--lr", "2e-6", "--warmup-steps", "10")
    flags = _sc_flags()
    flags[flags.index("--prefix-groups") + 1] = "68"
    a = D.parse_args(base + flags + PREFIX + ["--len-overlong-start", "0.8"])
    assert C.check_contract(a, "grpo_dapo_kl_sc_prefix_v2") == []
    assert (a.len_overlong_start, a.len_overlong_p0, a.len_overlong_p1, a.len_overlong_mixed_only) == (0.8, 0.05, 0.2, 1)
    with pytest.raises(SystemExit):
        D.parse_args(base + flags + PREFIX)                          # no penalty: not this contract


def test_grpo_dapo_clip_v1_contract():
    base = _base("grpo_dapo_clip_v1", "--lr", "1e-5", "--warmup-steps", "20")
    flags = list(CONTRACT_FLAGS)
    flags[flags.index("--prompts-per-step") + 1] = "272"
    flags[flags.index("--prefix-groups") + 1] = "68"
    flags[flags.index("--minibatch-max") + 1] = "2"
    pre = PREFIX + ["--pool-drop-solved", "1", "--kl-coef", "0.001", "--len-overlong-start", "0.8",
                    "--clip-eps-high", "0.28", "--ppo-old-logp", "trainer", "--adam-beta2", "0.99", "--weight-decay", "0.01"]
    a = D.parse_args(base + flags + pre)
    assert C.check_contract(a, "grpo_dapo_clip_v1") == []
    assert (a.pg_loss, a.clip_eps, a.clip_eps_high, a.minibatch_max) == ("ppo", 0.2, 0.28, 2)
    with pytest.raises(SystemExit):
        D.parse_args(base + flags + pre[:-2])                          # no weight decay: not this contract
    snap = C.objective_snapshot(a)["values"]
    assert snap["adam_beta2"] == 0.99 and snap["weight_decay"] == 0.01 and snap["ppo_old_logp"] == "trainer"
    with pytest.raises(SystemExit):                                    # trainer old log-probs only for the PPO loss
        D.parse_args(_base("grpo_dapo_kl_sc_prefix_v2") + _sc_flags() + ["--ppo-old-logp", "trainer"])


def test_force_suffix_env_override(monkeypatch):
    tok = lambda text, add_special_tokens=False: {"input_ids": [ord(c) for c in text]}   # noqa: E731
    monkeypatch.delenv("ARC_EVAL_FORCE_SUFFIX", raising=False)
    default, _ = D.force_suffix(tok)
    monkeypatch.setenv("ARC_EVAL_FORCE_SUFFIX", "\\n\\nNow final.\\n</think>\\n\\n")
    suf, ids = D.force_suffix(tok)
    assert suf == "\n\nNow final.\n</think>\n\n" and ids == [ord(c) for c in suf] and suf != default
    assert C.objective_snapshot(NS())["values"]["force_suffix_env"] == "\\n\\nNow final.\\n</think>\\n\\n"


# ---- pool order: dropped sources, solved tasks ------------------------------------------------------------------

def test_resume_with_a_dropped_source_starts_a_fresh_order():
    """A resume over a smaller pool (a source dropped) with --new-pool-on-resume starts a fresh order (the old indices
    would point at other tasks); without it the weighted order refuses to load."""
    old = rp.WeightedOrder([1] * 6, 0)
    old.take(4)
    state = {"args": {"pool_order": "weighted", "pool_file": None}, "order": old.state_dict()}
    new = rp.WeightedOrder([1] * 5, 0)
    logs = []
    args = types.SimpleNamespace(pool_order="weighted", new_pool_on_resume=True, pool_file=None, pool_extend=False)
    D.restore_prompt_order(args, state, new, logs.append, pool=None, start_step=3, pool_changed=True)
    assert any("fresh prompt order" in m for m in logs)
    assert new.order.state_dict() == rp.WeightedOrder([1] * 5, 0).order.state_dict()
    args.new_pool_on_resume = False
    with pytest.raises(ValueError):
        D.restore_prompt_order(args, state, rp.WeightedOrder([1] * 5, 0), logs.append, pool=None, start_step=3,
                               pool_changed=True)


def _write_groups(d, step, groups):
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / f"step_{step:06d}.jsonl.gz", "wt") as f:
        for g, (tid, outs) in enumerate(groups):
            for x in outs:
                f.write(json.dumps({"group": g, "task_id": tid, "seq": {"outcome": x}}) + "\n")


def test_drop_solved_order_and_record_seed(tmp_path):
    """--pool-drop-solved: solved items are never drawn again (also carried ones), the set survives state_dict /
    load_state_dict, and solved_from_records finds all-correct groups of the right size up to the resume step."""
    o = rp.WeightedOrder([1] * 10, 3)
    assert o.mark_solved([2, 5, 5]) == 2 and o.mark_solved([5]) == 0
    seen = [i for _ in range(6) for i in o.take(4)]
    assert 2 not in seen and 5 not in seen and set(seen) == set(range(10)) - {2, 5}
    st = o.state_dict()
    o2 = rp.WeightedOrder([1] * 10, 3)
    o2.load_state_dict(st)
    assert o2.solved == {2, 5} and o2.take(3) == o.take(3)
    o3 = rp.WeightedOrder([1] * 3, 0)
    o3.mark_solved([0, 1, 2])
    with pytest.raises(RuntimeError):
        o3.take(1)
    d = tmp_path / "rollouts"
    _write_groups(d, 1, [("a", [1] * 6), ("b", [1, 1, 1, 1, 1, 0]), ("gone", [1] * 6)])
    _write_groups(d, 2, [("c", [1] * 6), ("d", [1] * 5)])          # d: 5 samples, not a full group
    _write_groups(d, 4, [("e", [1] * 6)])                           # after the resume step: ignored
    idx = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    assert D.solved_from_records(str(tmp_path), 3, idx, 6) == {0, 2}


def test_drop_solved_seed_from_another_run(tmp_path):
    _write_groups(tmp_path / "old" / "rollouts", 2, [("a", [1] * 6), ("b", [1] * 5 + [0])])
    a = D.parse_args(["--out", str(tmp_path / "new"), "--student", "/tmp/none", "--pool-drop-solved", "1",
                      "--pool-drop-solved-seed", f"{tmp_path / 'old'}:10"])
    ((sdir, upto),) = D.parse_solved_seed(a.pool_drop_solved_seed)
    assert D.solved_from_records(sdir, upto, {"a": 0, "b": 1}, 6) == {0}


def test_drop_solved_seed_from_two_runs(tmp_path):
    """A fresh restart seeded with the solved sets of two earlier runs, each up to its own step."""
    for run, step, groups in (("v1", 10, [("a", [1] * 6), ("b", [1] * 5 + [0])]),
                              ("v1", 11, [("c", [1] * 6)]),                      # after UPTO: ignored
                              ("v2", 3, [("b", [1] * 6), ("a", [1] * 6)])):
        _write_groups(tmp_path / run / "rollouts", step, groups)
    spec = f"{tmp_path / 'v1'}:10,{tmp_path / 'v2'}:3"
    a = D.parse_args(["--out", str(tmp_path / "new"), "--student", "/tmp/none", "--pool-drop-solved", "1",
                      "--pool-drop-solved-seed", spec])
    specs = D.parse_solved_seed(a.pool_drop_solved_seed)
    assert specs == [(str(tmp_path / "v1"), 10), (str(tmp_path / "v2"), 3)]
    found = set()
    for sdir, upto in specs:
        found |= D.solved_from_records(sdir, upto, {"a": 0, "b": 1, "c": 2}, 6)
    assert found == {0, 1}
    with pytest.raises(SystemExit):
        D.parse_solved_seed(f"{tmp_path / 'v1'}:ten")
    with pytest.raises(SystemExit):
        D.parse_solved_seed(f"{tmp_path / 'missing'}:3")
    D.parse_solved_seed(f"{tmp_path / 'missing'}:3", check_dirs=False)      # the argument check only parses
    with pytest.raises(SystemExit):
        D.parse_args(["--out", str(tmp_path / "new"), "--student", "/tmp/none", "--pool-drop-solved-seed", spec])


# ---- multi-node minibatch plans --------------------------------------------------------------------------------

def test_node_affine_plan():
    """With node services every rollout whose top-k file sits under <topk_dir>/step_N/n<node>/ is trained by a rank of
    that node."""
    pytest.importorskip("torch")
    from arcsolver.rl import fsdp as F
    kept = []
    for i in range(12):
        node = i % 2
        kept.append({"group": i // 6, "prompt_ids": [0] * 10, "completion": [0] * (100 + 37 * i),
                     "topk_path": f"/dev/shm/rl_topk_x/step_000007/n{node}/s{i}_p{i}_s0.npz"})
    work = list(range(12))
    norms = {j: 1000.0 for j in work}
    plan = D.plan_minibatches(kept, work, 8, norms, "token", 1, 8, F.assign_equal, local_ws=4)
    assert len(plan) == 1
    for r, items in enumerate(plan[0]["assign"]):
        assert all(D.topk_node(kept[j]) == r // 4 for j in items), (r, items)
    assert sorted(j for a_ in plan[0]["assign"] for j in a_) == work
    single = D.plan_minibatches(kept, work, 8, norms, "token", 1, 8, F.assign_equal)        # no services: anywhere
    assert sorted(j for a_ in single[0]["assign"] for j in a_) == work
    assert D.topk_node({"topk_path": "/dev/shm/rl_topk_x/step_000001/s0_p0_s0.npz"}) is None


def test_remote_engine_topk_dir_is_per_node():
    c = workers.RemoteEngineClient.__new__(workers.RemoteEngineClient)
    c.service = types.SimpleNamespace(node_rank=1)
    assert c._node_topk({"topk_dir": "/dev/shm/rl_topk_r/step_000005", "x": 1}) == {
        "topk_dir": "/dev/shm/rl_topk_r/step_000005/n1", "x": 1}
    assert c._node_topk({"topk_dir": None, "x": 1}) == {"topk_dir": None, "x": 1}
    assert c._node_topk({"prompts": []}) == {"prompts": []}


# ---- soft overlong penalty and LR ramp (experimental options) ----------------------------------------------------

def test_overlong_penalty_dapo_mixed_only():
    """0 up to 80% of the cap, -0.05 -> -0.2 linearly to the cap, -0.2 truncated; mixed groups only."""
    cap = 1000
    vals = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    lens = [900, 800, 1000, 500, 999, 850, 950, 990, 700]
    trunc = [False, False, True, False, False, False, False, False, False]
    gids = [0, 0, 0, 0, 1, 1, 1, 2, 2]        # group 0 mixed, group 1 all wrong, group 2 all right
    out, n, g, tot = rr.overlong_penalty(vals, lens, trunc, vals, gids, cap, 0.8, 0.05, 0.2)
    assert out[0] == pytest.approx(1.0 - (0.05 + 0.15 * 0.5))        # 900 = half way 800..1000
    assert out[1] == 0.0                                               # exactly at the start: no penalty
    assert out[2] == pytest.approx(-0.2)                               # truncated
    assert out[3] == 1.0                                               # short
    assert out[4:7] == [0.0, 0.0, 0.0] and out[7:] == [1.0, 1.0]       # not mixed: untouched
    assert (n, g) == (2, 1) and tot == pytest.approx(0.125 + 0.2)
    out2, n2, _, _ = rr.overlong_penalty(vals, lens, trunc, vals, gids, cap, 0.8, 0.05, 0.2, mixed_only=False)
    assert n2 == 6 and out2[4] == pytest.approx(-(0.05 + 0.15 * 199 / 200))
    adv = rr.group_advantages(out, gids, "std", 1e-4)
    assert adv[0] < adv[3] and adv[2] < adv[1]                         # long right < short right; capped < wrong
    with pytest.raises(ValueError):
        rr.overlong_penalty(vals, lens, trunc, vals, gids, cap, 1.0, 0.05, 0.2)


def test_overlong_penalty_through_rollout_step(monkeypatch):
    """--len-overlong-start in rollout_step: truncated rollouts of mixed groups get -p1 (before the advantages), all-wrong
    groups are untouched (and stay dropped), the step's times carry the counts."""
    a, pool, step = rollout_harness(monkeypatch, ["--len-overlong-start", "0.8", "--prompts-per-step", "6"])
    order = RT.make_order(a, pool, [1] * len(pool))
    rollouts, kept, gaugs, prompts, res, times, _ = step(order, 1)
    by = {}
    for r in rollouts:
        by.setdefault(r["item"].task_id, []).append(r)
    four = by["t4"]                                    # 1 right, 3 truncated: mixed
    assert sum(r["overlong_pen"] == pytest.approx(0.2) for r in four) == 3
    assert all(r["r_seq"] == pytest.approx(-0.2) for r in four if r["finish"] == "length")
    assert max(r["seq_adv"] for r in four) > 0 > min(r["seq_adv"] for r in four)
    three = by["t3"]                                   # all wrong, one truncated: not penalised, dropped
    assert all(r["overlong_pen"] == 0.0 and r["r_seq"] == 0.0 and r["seq_adv"] == 0.0 for r in three)
    assert times["overlong"]["overlong/rollouts"] == 3 and times["overlong"]["overlong/groups"] == 1


def test_lr_warmup_and_ramp():
    """--lr after a linear warmup; with --lr-ramp-to linear from --lr to the target over the steps after
    --lr-ramp-from."""
    a = NS(lr=2e-6, warmup_steps=10, rewarmup_steps=5, lr_ramp_to=4e-6, lr_ramp_from=39, lr_ramp_steps=10)
    assert D.step_lr(a, 5) == pytest.approx(1e-6)                      # warmup
    assert D.step_lr(a, 39) == pytest.approx(2e-6)
    assert D.step_lr(a, 40) == pytest.approx(2.2e-6)
    assert D.step_lr(a, 44) == pytest.approx(3.0e-6)
    assert D.step_lr(a, 49) == pytest.approx(4e-6) and D.step_lr(a, 80) == pytest.approx(4e-6)
    off = NS(lr=2e-6, warmup_steps=10, rewarmup_steps=5)
    assert D.step_lr(off, 1) == pytest.approx(2e-7) and D.step_lr(off, 45) == pytest.approx(2e-6)   # no ramp flags
    assert D.step_lr(off, 12, rewarm_from=10) == pytest.approx(2e-6 * 2 / 5)                       # weights-only resume
