"""An independent, brute-force re-implementation of the ARC rival check (tests only).

It shares nothing with ``pv2.arc.rivals`` except the transform functions themselves (and their
``out_shape`` / ``noop`` declarations, which ``test_arc_transforms`` checks against the functions):
no ``solve``, no forward-pass bookkeeping, no candidate filters.  Nested loops over every library
instance:

* singles: every instance consistent with every train pair, every wrap shift by -29..29 rows and
  columns, and every object-size rule at every size;
* fitted colour map after the identity, and after any instance when no pure colour map fits;
* compositions T2 . T1 over ALL instance pairs, counted only when both steps are observable (dropping
  either one changes some train output), the rule the module docstring of ``pv2.arc.rivals`` states;
  a fixed colour-map T2 (recolor / swap / keep_fill / fill_const) is skipped when a pure colour map
  fits every pair, as the fitted map it stands for is.

``predictions(train, tests)`` returns, per test input, the set of predicted grids (as keys) and
whether a fitted map was ambiguous.
"""
from __future__ import annotations

import numpy as np

from pv2.arc.grids import Info, arr, fits
from pv2.arc.transforms import FAMILIES

INSTANCES = [(name, p) for name, fam in FAMILIES.items() for p in fam.instances]


def key(a):
    return None if a is None else (a.shape, a.tobytes())


def app(inst, g):
    if inst is None:
        return g
    fam = FAMILIES[inst[0]]
    r = fam.fn(Info(g), *inst[1])
    if r is None or not fits(*r.shape):
        return None
    return r


def same(a, b):
    return a is not None and b is not None and a.shape == b.shape and bool(np.array_equal(a, b))


def cmap(z, y):
    if z.shape != y.shape:
        return None
    m = {}
    for a, b in zip(z.ravel().tolist(), y.ravel().tolist()):
        if m.setdefault(a, b) != b:
            return None
    return m


def merge_maps(zs, ys):
    m = {}
    for z, y in zip(zs, ys):
        c = cmap(z, y)
        if c is None:
            return None
        for a, b in c.items():
            if m.setdefault(a, b) != b:
                return None
    return m


def predictions(train, tests, *, compositions: bool = True):
    xs = [arr(x) for x, _ in train]
    ys = [arr(y) for _, y in train]
    ts = [arr(t) for t in tests]
    preds = [set() for _ in ts]
    ambiguous = [False for _ in ts]
    tokens = []

    single_fits = {}
    for inst in INSTANCES:
        ok = all(same(app(inst, x), y) for x, y in zip(xs, ys))
        single_fits[inst] = ok
        if ok:
            tokens.append(inst)
            for j, t in enumerate(ts):
                p = app(inst, t)
                if p is not None:
                    preds[j].add(key(p))

    # every wrap shift by -29..29 rows and columns, as a single step
    if all(x.shape == y.shape for x, y in zip(xs, ys)):
        for dr in range(-29, 30):
            for dc in range(-29, 30):
                if all(np.array_equal(np.roll(x, (dr, dc), axis=(0, 1)), y) for x, y in zip(xs, ys)):
                    tokens.append(("roll_any", (dr, dc)))
                    for j, t in enumerate(ts):
                        preds[j].add(key(np.ascontiguousarray(np.roll(t, (dr, dc), axis=(0, 1)))))

    # object-size rules with any size: the outcome only changes at the object sizes present, so n at
    # every size s and s+1 of the train and test inputs (all colours, both connectivities)
    sizes = set()
    for g in xs + ts:
        info = Info(g)
        for conn in (4, 8):
            objs = info.objects(conn)
            if objs:
                sizes |= {o.size for o in objs}
    for n in sorted(sizes | {s + 1 for s in sizes}):
        for conn in (4, 8):
            insts = [("remove_size", (cmp, n, conn)) for cmp in ("eq", "lt", "gt")]
            for x in range(10):
                insts.append(("recolor_size", (n, x, conn)))
                insts += [("recolor_size_cmp", (cmp, n, x, conn)) for cmp in ("lt", "gt")]
            for inst in insts:
                if inst in single_fits:
                    continue
                if all(same(app(inst, x), y) for x, y in zip(xs, ys)):
                    tokens.append(inst)
                    for j, t in enumerate(ts):
                        p = app(inst, t)
                        if p is not None:
                            preds[j].add(key(p))

    # first steps grouped by their results on every train input and every test input
    groups = {}
    for inst in [None] + INSTANCES:
        zs = [app(inst, x) for x in xs]
        if any(z is None for z in zs):
            continue
        zt = [app(inst, t) for t in ts]
        k = tuple(key(z) for z in zs) + tuple(key(z) for z in zt)
        g = groups.get(k)
        if g is None:
            groups[k] = g = (zs, zt, [])
        g[2].append(inst)

    pure = merge_maps(xs, ys) is not None
    for zs, zt, firsts in groups.values():
        if not (None in firsts or not pure):
            continue
        m = merge_maps(zs, ys)
        if m is None:
            continue
        for j, z in enumerate(zt):
            if z is None:
                continue
            if any(int(c) not in m for c in np.unique(z)):
                ambiguous[j] = True
            else:
                lut = np.arange(10, dtype=np.int8)
                for a, b in m.items():
                    lut[a] = b
                preds[j].add(key(lut[z]))

    if compositions:
        # second steps are tried once per distinct first-pair intermediate (one shared Info, so
        # per-grid analysis is reused); instances whose declared result size, no-op test or colour
        # set rule them out are skipped (those declarations are checked in test_arc_transforms)
        by_z0 = {}
        for zs, zt, firsts in groups.values():
            if firsts == [None]:
                continue
            if all(same(z, x) for z, x in zip(zs, xs)):
                continue          # the first step never acts: dropping it changes nothing
            if all(same(z, y) for z, y in zip(zs, ys)):
                continue          # the first step alone fits: any second step is unobservable
            by_z0.setdefault(key(zs[0]), []).append((zs, zt, firsts))
        y0 = ys[0]
        ycols = set(np.unique(y0).tolist())
        for glist in by_z0.values():
            z0 = glist[0][0][0]
            info0 = Info(z0)
            known = ycols <= set(np.unique(z0).tolist())
            second = []
            for inst2 in INSTANCES:
                fam = FAMILIES[inst2[0]]
                if not known and not fam.new_colors:
                    continue
                if pure and fam.sig == "fixed_map":
                    # a fixed recolour / swap / keep-and-fill after a step is what the fitted map stands
                    # for, and a fitted map's first step is dropped when a pure colour map fits (the
                    # rivals docstring); the fitted-map branch above skips the same rivals
                    continue
                if not fam.same_shape:
                    s = fam.out_shape(z0.shape[0], z0.shape[1], inst2[1])
                    if s is False or (s is not None and s != y0.shape):
                        continue
                elif z0.shape != y0.shape:
                    continue
                if fam.noop is not None and fam.noop(info0, *inst2[1]):
                    continue
                r = fam.fn(info0, *inst2[1])
                if r is not None and fits(*r.shape) and same(r, y0):
                    second.append(inst2)
            for zs, zt, firsts in glist:
                for inst2 in second:
                    if not all(same(app(inst2, z), y) for z, y in zip(zs[1:], ys[1:])):
                        continue
                    if single_fits[inst2]:
                        continue  # the second step alone fits: the first is unobservable
                    tokens.append((firsts[0], inst2))
                    for j, z in enumerate(zt):
                        if z is None:
                            continue
                        p = app(inst2, z)
                        if p is not None:
                            preds[j].add(key(p))
    return preds, ambiguous, tokens
