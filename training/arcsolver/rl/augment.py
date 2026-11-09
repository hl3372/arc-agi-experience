"""Task augmentation for RL (pure python): a dihedral transform composed with a colour permutation, applied to EVERY
grid of a task (train inputs and outputs, test inputs and the test outputs = the ground truth), so the augmented task
is a valid ARC task whose correct answers are exactly the transformed correct answers of the original.

Why: most RL "solves" on the unaugmented pool were lucky guesses on small-answer tasks or recalled SFT tasks. A fresh
random view of the task per rollout group removes recall and makes guesses much less likely to hit, so the outcome
reward has to be earned by the rule. The eval set stays unaugmented.

  D8 (grid H x W; new[i][j] = old[src]):
    id              new[i][j] = g[i][j]                 H x W
    rot90           new[i][j] = g[H-1-j][i]             W x H   (clockwise)
    rot180          new[i][j] = g[H-1-i][W-1-j]         H x W
    rot270          new[i][j] = g[j][W-1-i]             W x H   (counter-clockwise)
    flip_h          new[i][j] = g[i][W-1-j]             H x W   (left-right mirror)
    flip_v          new[i][j] = g[H-1-i][j]             H x W   (upside down)
    transpose       new[i][j] = g[j][i]                 W x H
    anti_transpose  new[i][j] = g[H-1-j][W-1-i]         W x H
  colours: new colour = perm[old colour]; "nonzero" permutes 1-9 and keeps 0 (the usual background) fixed, "all"
  permutes 0-9, "none" keeps them. Colour maps and dihedral maps commute.

An Aug is sampled once per rollout group (all G samples share it, so group normalisation compares like with like) from
a seeded RNG (seed, step, group) -> reproducible across resumes.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass

D8 = ("id", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose")
INVERSE = {"id": "id", "rot90": "rot270", "rot180": "rot180", "rot270": "rot90", "flip_h": "flip_h",
           "flip_v": "flip_v", "transpose": "transpose", "anti_transpose": "anti_transpose"}
COLOR_MODES = ("nonzero", "all", "none")


def _transpose(g):
    return [list(r) for r in zip(*g)]


def dihedral(g, name: str):
    """One of the 8 symmetries of the rectangle (see the module table)."""
    if name == "id":
        return [list(r) for r in g]
    if name == "rot90":
        return [list(r) for r in zip(*g[::-1])]
    if name == "rot180":
        return [list(r[::-1]) for r in g[::-1]]
    if name == "rot270":
        return [list(r) for r in zip(*g)][::-1]
    if name == "flip_h":
        return [list(r[::-1]) for r in g]
    if name == "flip_v":
        return [list(r) for r in g[::-1]]
    if name == "transpose":
        return _transpose(g)
    if name == "anti_transpose":
        return [list(r[::-1]) for r in _transpose(g)[::-1]]
    raise ValueError(f"unknown dihedral transform {name!r}")


@dataclass(frozen=True)
class Aug:
    d8: str = "id"
    perm: tuple = tuple(range(10))        # new colour = perm[old colour]

    def is_identity(self) -> bool:
        return self.d8 == "id" and self.perm == tuple(range(10))

    def inverse(self) -> "Aug":
        inv = [0] * 10
        for old, new in enumerate(self.perm):
            inv[new] = old
        return Aug(INVERSE[self.d8], tuple(inv))

    def grid(self, g):
        return dihedral([[self.perm[v] for v in row] for row in g], self.d8)

    def task(self, task: dict) -> dict:
        """A deep copy of the task with every grid transformed (other keys are kept)."""
        out = copy.deepcopy(task)
        for split in ("train", "test"):
            for pair in out.get(split, []):
                for k in ("input", "output"):
                    if k in pair:
                        pair[k] = self.grid(pair[k])
                if "alt_outputs" in pair:              # other accepted test answers (optional task field)
                    pair["alt_outputs"] = [self.grid(g) for g in pair["alt_outputs"]]
        return out

    def meta(self) -> dict:
        return {"d8": self.d8, "perm": "".join(map(str, self.perm))}


IDENTITY = Aug()


def sample_aug(rng: random.Random, colors: str = "nonzero", identity_prob: float | None = None) -> Aug:
    """Dihedral: uniform over D8 (identity 1/8) or, with identity_prob set, identity with that probability and the 7
    others uniformly; colours: a uniform random permutation per `colors`."""
    if colors not in COLOR_MODES:
        raise ValueError(colors)
    if identity_prob is None or identity_prob < 0:
        d8 = rng.choice(D8)
    else:
        d8 = "id" if rng.random() < identity_prob else rng.choice(D8[1:])
    if colors == "none":
        perm = tuple(range(10))
    elif colors == "nonzero":
        p = list(range(1, 10))
        rng.shuffle(p)
        perm = tuple([0] + p)
    else:
        p = list(range(10))
        rng.shuffle(p)
        perm = tuple(p)
    return Aug(d8, perm)


def group_rng(seed: int, step: int, group: int) -> random.Random:
    """Per (run seed, step, group) RNG: the same step of a resumed run draws the same augmentations."""
    return random.Random((seed * 1_000_003 + step) * 10_007 + group + 7_777_777)
