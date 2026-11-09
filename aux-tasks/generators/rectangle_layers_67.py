"""Readable, uniquely recoverable layers of filled rectangles for task 67.

Dimensions, group sizes, palette and layout families are planned before retries.
The independent reader uses visible colors and occlusion constraints; it does
not receive the construction rectangles or their order.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


class InvalidScene(ValueError):
    """The visible grid does not certify one rectangle scene and reversed output."""


@dataclass
class Reading:
    background: object
    rectangles: dict
    groups: list
    output: np.ndarray
    overlap_cells: int
    common_core_groups: int


def _bounds(grid, color):
    rows, cols = np.nonzero(grid == color)
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _paint(shape, bg, rectangles, order):
    out = np.full(shape, bg)
    for color in order:
        a, b, c, d = rectangles[color]
        out[a:c, b:d] = color
    return out


def _read_field(grid, bg, *, certify_extents=True):
    colors = [c.item() for c in np.unique(grid) if c != bg]
    if not 2 <= len(colors) <= 9:
        raise InvalidScene("need at least two visible rectangles")
    rectangles = {color: _bounds(grid, color) for color in colors}
    n = len(colors)
    index = {color: i for i, color in enumerate(colors)}
    before = [0] * n
    masks = []
    for i, color in enumerate(colors):
        a, b, c, d = rectangles[color]
        if c - a < 2 or d - b < 2:
            raise InvalidScene("a rectangle must have two nondegenerate sides")
        mask = np.zeros(grid.shape, dtype=bool)
        mask[a:c, b:d] = True
        masks.append(mask)
        for cell in np.unique(grid[mask]):
            if cell == bg:
                raise InvalidScene("background visible through a filled rectangle")
            if cell != color:
                before[index[cell.item()]] |= 1 << i
    # Transitive closure of all visibly required back-to-front relations.
    for k in range(n):
        for j in range(n):
            if before[j] & (1 << k):
                before[j] |= before[k]
    if any(before[j] & (1 << j) for j in range(n)):
        raise InvalidScene("inconsistent layer order")
    h, w = grid.shape
    if certify_extents:
        for i, color in enumerate(colors):
            a, b, c, d = rectangles[color]
            strips = []
            if a: strips.append(grid[a - 1, b:d])
            if c < h: strips.append(grid[c, b:d])
            if b: strips.append(grid[a:c, b - 1])
            if d < w: strips.append(grid[a:c, d])
            for strip in strips:
                cells = np.unique(strip)
                if bg in cells:
                    continue
                # Extending this rectangle would require it to lie below every
                # color in the new strip. An already lower color contradicts it.
                if not any(before[i] & (1 << index[x.item()]) for x in cells):
                    raise InvalidScene("a hidden rectangle extension remains possible")
        # This one-step test covers arbitrary hidden enlargements: any larger
        # filled rectangle contains an immediate outward strip. Shrinking all
        # other rectangles to their visible boxes preserves the observed image.
    masks = np.asarray(masks)
    out = np.full(grid.shape, bg, dtype=grid.dtype)
    winner_count = np.zeros(grid.shape, dtype=np.uint8)
    for i, color in enumerate(colors):
        earlier = [j for j in range(n) if before[i] & (1 << j)]
        blocked = np.any(masks[earlier], axis=0) if earlier else False
        wins = masks[i] & ~blocked if earlier else masks[i]
        out[wins] = color
        winner_count += wins
    if np.any(winner_count > 1):
        raise InvalidScene("compatible orders give different reversed pixels")
    groups = []
    unseen = set(range(n))
    while unseen:
        component = {unseen.pop()}
        todo = list(component)
        while todo:
            i = todo.pop()
            for j in list(unseen):
                if np.any(masks[i] & masks[j]):
                    unseen.remove(j)
                    component.add(j)
                    todo.append(j)
        groups.append([colors[j] for j in sorted(component)])
    overlap = int(np.sum(masks.sum(axis=0) > 1))
    if not overlap or np.array_equal(out, grid):
        raise InvalidScene("there must be a visible layer reversal")
    common = sum(bool(np.all(masks[[index[c] for c in group]], axis=0).any())
                 for group in groups if len(group) > 1)
    return Reading(bg, rectangles, groups, out, overlap, common)


def read_scene(grid):
    grid = np.asarray(grid)
    if grid.ndim != 2 or not all(5 <= n <= 30 for n in grid.shape):
        raise InvalidScene("canvas sides must be 5 through 30")
    explanations = []
    for color in np.unique(grid):
        try:
            explanations.append(_read_field(grid, color.item()))
        except InvalidScene:
            continue
    if len(explanations) != 1:
        raise InvalidScene("the background and visible rectangle scene are not unique")
    return explanations[0]


def validate(example):
    reading = read_scene(example['input'])
    if not np.array_equal(reading.output, np.asarray(example['output'])):
        raise InvalidScene("output disagrees with independently inferred layers")
    return reading


def _rand_rect(rng, h, w, *, broad=False):
    rh = int(rng.integers(max(2, h // 2) if broad else 2, h + 1))
    rw = int(rng.integers(max(2, w // 2) if broad else 2, w + 1))
    r = int(rng.integers(h - rh + 1)); c = int(rng.integers(w - rw + 1))
    return r, c, r + rh, c + rw


def _group_candidates(rng, h, w, n, family):
    if family == 'nested':
        rects = [(0, 0, h, w)]
        for _ in range(n - 1):
            a, b, c, d = _rand_rect(rng, h - 2, w - 2)
            rects.append((a + 1, b + 1, c + 1, d + 1))
        return rects  # the containing rectangle must start behind the others
    if family == 'cross':
        # Rectangles crossing at right angles, with visible ends on both axes.
        th = int(rng.integers(2, h - 1))
        tw = int(rng.integers(2, w - 1))
        rr = int(rng.integers(1, h - th))
        cc = int(rng.integers(1, w - tw))
        rects = [(rr, 0, rr + th, w), (0, cc, h, cc + tw)]
        for _ in range(n - 2):
            rects.append(_rand_rect(rng, h, w))
        rng.shuffle(rects)
        return rects
    if family == 'chain':
        # A diagonal sequence: local intersections, no common core for 3+.
        dy = max(1, (h - 2) // (n - 1))
        dx = max(1, (w - 2) // (n - 1))
        rh = max(2, h - (n - 1) * dy)
        rw = max(2, w - (n - 1) * dx)
        # Allocate more span to the overlapping bodies when there is room.
        if dy > 1:
            dy = int(rng.integers(max(1, dy // 2), dy + 1))
            rh = min(h - (n - 1) * dy, int(rng.integers(dy + 1, 2 * dy + 1)))
        if dx > 1:
            dx = int(rng.integers(max(1, dx // 2), dx + 1))
            rw = min(w - (n - 1) * dx, int(rng.integers(dx + 1, 2 * dx + 1)))
        rects = [(i * dy, i * dx, i * dy + rh, i * dx + rw) for i in range(n)]
        rng.shuffle(rects)
        return rects
    if family == 'offset' and n == 2:
        # Independently sized corner offsets retain full visible extents.
        a = int(rng.integers(2, h)); b = int(rng.integers(2, w))
        c = int(rng.integers(1, a)); d = int(rng.integers(1, b))
        rects = [(0, 0, a, b), (c, d, h, w)]
        rng.shuffle(rects)
        return rects
    rects = [_rand_rect(rng, h, w, broad=True) for _ in range(n)]
    rng.shuffle(rects)
    return rects


def _group(rng, h, w, n, family):
    for attempt in range(1, 401):
        rects = _group_candidates(rng, h, w, n, family)
        if any(c-a < 2 or d-b < 2 for a,b,c,d in rects):
            continue
        colors = list(range(1, n + 1))
        boxes = {color: (a+1,b+1,c+1,d+1) for color,(a,b,c,d) in zip(colors,rects)}
        inp = _paint((h+2,w+2),0,boxes,colors)
        if set(np.unique(inp)) != set(range(n+1)):
            continue
        if any(_bounds(inp,c) != boxes[c] for c in colors):
            continue
        try:
            reading = _read_field(inp,0)
        except InvalidScene:
            continue
        if len(reading.groups) != 1:
            continue
        if family == 'chain' and n > 2 and reading.common_core_groups:
            continue
        if family == 'stack' and not reading.common_core_groups:
            continue
        # Random spatial orientation without changing the already valid order.
        if rng.random() < .5:
            rects = [(h-c,b,h-a,d) for a,b,c,d in rects]
        if rng.random() < .5:
            rects = [(a,w-d,c,w-b) for a,b,c,d in rects]
        return rects, attempt
    return None, 400


def _place(rng, used, local_rects, h, w, margin):
    rows, cols = used.shape
    for _ in range(100):
        r = int(rng.integers(margin, rows-h-margin+1))
        c = int(rng.integers(margin, cols-w-margin+1))
        shifted = [(r+a,c+b,r+d,c+e) for a,b,d,e in local_rects]
        mask = np.zeros(used.shape,dtype=bool)
        for a,b,d,e in shifted: mask[a:d,b:e] = True
        if np.any(mask & used):
            continue
        # Two exterior corner witnesses keep the uniform field recognizable;
        # other corners and canvas edges can still be occupied.
        if mask[0,0] or mask[-1,-1]:
            continue
        blocked = mask.copy()
        blocked[1:] |= mask[:-1]; blocked[:-1] |= mask[1:]
        blocked[:,1:] |= mask[:,:-1]; blocked[:,:-1] |= mask[:,1:]
        used |= blocked
        return shifted
    return None


def _minimum_side(n, family):
    side = 3 if n == 2 else n + 1
    if family == 'cross': side = max(side, 4)
    if family == 'nested': side = max(side, 2*n)
    if family == 'stack': side = max(side, 5)
    return side


def _shelf_fits(h, w, sides):
    """Conservative footprint witness used before selecting a layout family."""
    for hh, ww in ((h,w),(w,h)):
        x = y = row_height = 0
        for side in sorted(sides,reverse=True):
            if side > ww: break
            if x and x + side > ww:
                y += row_height + 1; x = row_height = 0
            if y + side > hh: break
            x += side + 1; row_height = max(row_height,side)
        else:
            return True
    return False


def generate(rng):
    h,w = map(int,rng.integers(5,31,size=2))
    margin = 1 if rng.random() < .7 else 0
    ah,aw = h-2*margin,w-2*margin
    area = ah*aw
    max_groups = min(3,max(1,(max(ah,aw))//6),max(1,area//90))
    probs = np.asarray([.62,.30,.08][:max_groups]);probs /= probs.sum()
    n_groups = int(rng.choice(np.arange(1,max_groups+1),p=probs))
    # Four layers need enough surrounding field to keep their visible remnants
    # readable. This is capacity planning, before the layer count is selected.
    short_side = min(ah,aw)
    max_layers = 2 if short_side < 4 else 3 if short_side < 7 else 4
    sizes=[]
    for i in range(n_groups):
        cap = min(max_layers,9-sum(sizes)-2*(n_groups-i-1))
        weights=np.asarray([.52,.31,.17][:cap-1]);weights/=weights.sum()
        sizes.append(int(rng.choice(np.arange(2,cap+1),p=weights)))
    base_sides=[3 if n==2 else n+1 for n in sizes]
    extra_options=[k for k in range(min(3,9-sum(sizes),area//65)+1)
                   if _shelf_fits(ah,aw,base_sides+[2]*k)]
    extras = int(rng.choice(extra_options))
    families=[]
    for n in sizes:
        choices = ['offset'] if n==2 else ['chain']
        weights = [4.]
        if min(ah,aw)>=4:
            choices.append('cross');weights.append(3. if n==2 else 1.)
        if min(ah,aw)>=max(4,2*n):
            choices.append('nested');weights.append(2.)
        if n>=3 and min(ah,aw)>=max(5,n+1):
            choices.append('stack');weights.append(3.)
        i=len(families)
        previous=[_minimum_side(s,f) for s,f in zip(sizes,families)]
        eligible=[j for j,f in enumerate(choices)
                  if _shelf_fits(ah,aw,previous+[_minimum_side(n,f)]+base_sides[i+1:]+[2]*extras)
                  and (f!='nested' or _minimum_side(n,f)**2 <= area*.55)]
        choices=[choices[j] for j in eligible];weights=[weights[j] for j in eligible]
        probs=np.asarray(weights)/sum(weights)
        families.append(str(rng.choice(choices,p=probs)))
    palette=[int(c) for c in rng.choice(10,size=1+sum(sizes)+extras,replace=False)]
    bg=palette[0]
    for attempt in range(1,201):
        used=np.zeros((h,w),dtype=bool)
        grouped=[];inner_attempts=[];failed=False
        for n,family in zip(sizes,families):
            min_side = _minimum_side(n,family)
            target = area*float(rng.uniform(.35,.65))/(n_groups+.3*extras)
            aspect = float(rng.lognormal(0,.55))
            gh=min(ah,max(min_side,int(round(math.sqrt(target*aspect)))))
            gw=min(aw,max(min_side,int(round(target/gh))))
            local,tries=_group(rng,gh,gw,n,family)
            inner_attempts.append(tries)
            if local is None:
                failed=True;break
            placed=_place(rng,used,local,gh,gw,margin)
            if placed is None:
                failed=True;break
            grouped.append(placed)
        if failed:continue
        isolated=[]
        for _ in range(extras):
            rh=int(rng.integers(2,max(3,min(ah,round(math.sqrt(area)*.35))+1)))
            rw=int(rng.integers(2,max(3,min(aw,round(math.sqrt(area)*.35))+1)))
            placed=_place(rng,used,[(0,0,rh,rw)],rh,rw,margin)
            if placed is None:failed=True;break
            isolated.extend(placed)
        if failed:continue
        rectangles={}; order=[]; reverse_order=[];idx=1
        for group in grouped:
            group_colors=palette[idx:idx+len(group)];idx+=len(group)
            rectangles.update(zip(group_colors,group));order+=group_colors
            reverse_order+=group_colors[::-1]
        for box in isolated:
            color=palette[idx];idx+=1;rectangles[color]=box
            order.append(color);reverse_order.append(color)
        inp=_paint((h,w),bg,rectangles,order)
        counts=np.array([(inp==c).sum() for c in palette])
        boundary=np.concatenate((inp[0],inp[-1],inp[1:-1,0],inp[1:-1,-1]))
        if counts[0] < counts[1:].max()*1.1 or np.mean(boundary==bg)<.35:
            continue
        out=_paint((h,w),bg,rectangles,reverse_order)
        example={'input':inp.tolist(),'output':out.tolist()}
        try:
            reading=validate(example)
        except InvalidScene:
            continue
        actual=sorted(len(g) for g in reading.groups if len(g)>1)
        if actual!=sorted(sizes) or sum(len(g)==1 for g in reading.groups)!=extras:
            continue
        n_rectangles=len(rectangles)
        visible=[int((inp==c).sum()) for c in order]
        extents=[(rectangles[c][2]-rectangles[c][0],rectangles[c][3]-rectangles[c][1]) for c in order]
        example['metadata']={
            'grid_h':h,'grid_w':w,'n_colors':len(palette),'background':bg,
            'background_margin':float(counts[0]/counts[1:].max()),
            'background_boundary_share':float(np.mean(boundary==bg)),
            'texture_group':None,'texture_kind':'uniform','texture_named':False,
            'texture_colors':[],'texture_coverage':0.,
            'n_rectangles':n_rectangles,'n_overlap_groups':n_groups,
            'overlap_group_sizes':actual,'max_overlap_group_size':max(actual),
            'n_extra':extras,'planned_group_sizes':sizes,'planned_families':families,
            'rectangles':[{'color':c,'box':list(rectangles[c]),'visible_area':int((inp==c).sum())} for c in order],
            'max_rectangle_area':max(a*b for a,b in extents),
            'max_rectangle_side':max(max(v) for v in extents),
            'occupancy':float(np.mean(inp!=bg)),
            'overlap_cells':reading.overlap_cells,
            'common_core_groups':reading.common_core_groups,
            'non_common_core_groups':n_groups-reading.common_core_groups,
            'edge_contact':bool(np.any(boundary!=bg)),
            'attempts':attempt,'group_attempts':inner_attempts,
            'extent_certified':True,'reversed_output_unique':True,
        }
        return example
    raise RuntimeError(f'task67 could not realize fixed plan {(h,w,sizes,extras,families)}')
