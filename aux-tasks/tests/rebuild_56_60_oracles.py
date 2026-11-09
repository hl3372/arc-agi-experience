"""Written from rendered prompts and visible examples before reading production solvers.
Exact and family searches use the explicitly stated whole 4-component rule.
"""
import re
from collections import Counter
import numpy as np

D4=[(-1,0),(1,0),(0,-1),(0,1)]
D8=[(r,c) for r in (-1,0,1) for c in (-1,0,1) if r or c]

def components(mask,conn=4):
    rem=set(map(tuple,np.argwhere(mask))); out=[]
    ds=D4 if conn==4 else D8
    while rem:
        p=rem.pop(); comp={p}; stack=[p]
        while stack:
            r,c=stack.pop()
            for dr,dc in ds:
                q=r+dr,c+dc
                if q in rem: rem.remove(q);comp.add(q);stack.append(q)
        out.append(comp)
    return out

def norm(cells):
    r0=min(p[0] for p in cells);c0=min(p[1] for p in cells)
    return frozenset((r-r0,c-c0) for r,c in cells)

def dims(cells):
    return (max(r for r,c in cells)-min(r for r,c in cells)+1,max(c for r,c in cells)-min(c for r,c in cells)+1)

def variants(cells,rule):
    v=set()
    for flip in range(2 if rule=='dihedral' else 1):
        s={(r,-c if flip else c) for r,c in cells}
        for turn in range(4 if rule!='exact' else 1):
            v.add(norm(s));s={(-c,r) for r,c in s}
    return v

def color(s,a):
    return s if a.dtype.kind in ('O','U','S') else int(s)

def mode(values):
    co=Counter(values)
    if not co:return None
    n=max(co.values());ws=[k for k,v in co.items() if v==n]
    return ws[0] if len(ws)==1 else None

def parse58(prompt):
    a,b=re.search(r'group .*?of color (\w+) cells.*?template ([.#/]+)',prompt).groups()
    tgt=re.search(r'Recolor every cell of each such group to color (\w+)',prompt).group(1)
    cells={(r,c) for r,line in enumerate(b.split('//')) for c,v in enumerate(line) if v=='#'}
    rule='dihedral' if 'reflected copies also count' in prompt else 'rotations' if 'rotated copies also count' in prompt else 'exact'
    return a,tgt,cells,rule

def solve56(grid,prompt):
    a=np.asarray(grid);oc,tc=re.search(r'exactly one color (\w+) object and exactly one color (\w+) target',prompt).groups();oc=color(oc,a);tc=color(tc,a)
    obj=set(map(tuple,np.argwhere(a==oc)));target=set(map(tuple,np.argwhere(a==tc)))
    adj4=sum((r+dr,c+dc) in target for r,c in obj for dr,dc in D4)
    adj8=sum((r+dr,c+dc) in target for r,c in obj for dr,dc in D8)
    md=min(max(abs(r-x),abs(c-y)) for r,c in obj for x,y in target)
    mode8='8-neighbor adjacent' in prompt
    out='yes' if (adj8 if mode8 else adj4) else 'no'
    return out,{'rule':8 if mode8 else 4,'object_area':len(obj),'target_area':len(target),'adj4':adj4,'adj8':adj8,'cheb_distance':md,'answer_changes_rule':bool(adj8 and not adj4),'object_comps4':len(components(a==oc)),'target_comps4':len(components(a==tc)),'object_comps8':len(components(a==oc,8)),'target_comps8':len(components(a==tc,8)),'objh':dims(obj)[0],'objw':dims(obj)[1],'tgth':dims(target)[0],'tgtw':dims(target)[1],'main_occupancy':(len(obj)+len(target))/a.size}

def solve57(grid,prompt):
    a=np.asarray(grid);h,w=a.shape;wall=color(re.search(r'wall of color (\w+)',prompt).group(1),a)
    allowed=set(map(tuple,np.argwhere(a!=wall)))
    stack=[p for p in allowed if p[0] in (0,h-1) or p[1] in (0,w-1)];outside=set(stack)
    while stack:
        r,c=stack.pop()
        for dr,dc in D4:
            q=r+dr,c+dc
            if q in allowed and q not in outside:outside.add(q);stack.append(q)
    inside=allowed-outside
    excluded=re.search(r'ignoring cells of color (\w+)',prompt)
    excluded=color(excluded.group(1),a) if excluded else None
    if 'ignoring the background' in prompt:
        excluded=mode(list(a.flat))
        if excluded is None:raise ValueError('ambiguous background')
    vals=[a[p] for p in inside if a[p]!=excluded];co=Counter(vals)
    answer=mode(vals)
    if answer is None:raise ValueError('non-unique or empty answer')
    im=np.zeros(a.shape,bool)
    for p in inside:im[p]=1
    wi=np.argwhere(a==wall);r0,c0=wi.min(0);r1,c1=wi.max(0)
    box=[v for v in a[r0+1:r1,c0+1:c1].flat if v!=wall and v!=excluded]
    global_=[v for v in a.flat if v!=wall and v!=excluded]
    ranked=sorted(co.values(),reverse=True)
    core={'ignore_bg':excluded is not None,'inside_area':len(inside),'inside_colors_all':len(set(a[p] for p in inside)),'inside_colors_counted':len(co),'inside_regions':len(components(im)),'winner_count':ranked[0],'margin':ranked[0]-(ranked[1] if len(ranked)>1 else 0),'winner_share':ranked[0]/len(vals),'global_modal_correct':mode(list(a.flat))==answer,'global_excluded_correct':mode(global_)==answer,'bbox_answer_correct':mode(box)==answer,'ignore_changes_answer':mode([a[p] for p in inside])!=answer,'inside_bbox_area':dims(inside)[0]*dims(inside)[1],'inside_occupancy':len(inside)/a.size,'first_inside_correct':a[min(inside)]==answer}
    return str(answer),core

def solve58(grid,prompt):
    a=np.asarray(grid);oc,tgt,template,rule=parse58(prompt);oc=color(oc,a);tgt=color(tgt,a)
    comps=components(a==oc); shapes=[norm(c) for c in comps];vs=variants(template,rule)
    flags=[s in vs for s in shapes];out=a.copy()
    for c,f in zip(comps,flags):
        if f:
            for p in c:out[p]=tgt
    exact=[s in variants(template,'exact') for s in shapes];rot=[s in variants(template,'rotations') for s in shapes];dih=[s in variants(template,'dihedral') for s in shapes]
    cheap_size=[len(s)==len(template) for s in shapes]
    cheap_box=[dims(s) in {dims(v) for v in vs} for s in shapes]
    comp8=components(a==oc,8);matched8=set().union(*(c for c in comp8 if norm(c) in vs)) if comp8 else set()
    matched=set().union(*(c for c,f in zip(comps,flags) if f)) if any(flags) else set()
    altcolors=0
    for col in np.unique(a):
        if col!=oc:altcolors+=sum(norm(c) in vs for c in components(a==col))
    return out.tolist(),{'rule':rule,'template_area':len(template),'template_h':dims(template)[0],'template_w':dims(template)[1],'n_target_components':len(comps),'n_matches':sum(flags),'n_target_cells':int((a==oc).sum()),'n_distractors':len(comps)-sum(flags),'size_only_correct':cheap_size==flags,'box_only_correct':cheap_box==flags,'all_target_correct':all(flags),'exact_only_correct':exact==flags,'rotations_only_correct':rot==flags,'dihedral_only_correct':dih==flags,'conn8_correct':matched8==matched,'other_color_matches':altcolors,'n_same_size_nearmisses':sum(q and not f for q,f in zip(cheap_size,flags)),'n_component_sizes':len({len(c) for c in comps}),'target_occupancy':int((a==oc).sum())/a.size}

def parse59(prompt):
    oc=re.search(r'color (\w+)',prompt).group(1)
    pat=re.search(r'template ([.#/]+)',prompt)
    rule='dihedral' if 'reflected copies also count' in prompt else 'rotations' if 'rotated copies also count' in prompt else 'exact'
    if pat:
        cells={(r,c) for r,row in enumerate(pat.group(1).split('//')) for c,v in enumerate(row) if v=='#'}
        return oc,'exact',cells,rule
    fam='rectangle' if 'non-square rectangle' in prompt else 'frame' if 'rectangular frame' in prompt else 'square' if 'filled square' in prompt else 'line' if 'straight line' in prompt else 'plus' if 'plus with' in prompt else re.search(r'one-cell-thick ([LTUH]) ',prompt).group(1)
    return oc,fam,None,'dihedral'

def generic59(cells,fam,rule,prompt):
    for s in variants(cells,'dihedral'):
        h,w=dims(s)
        if fam=='rectangle' and h!=w and min(h,w)>=2 and len(s)==h*w:return True
        if fam=='square' and h==w and len(s)==h*w:return True
        if fam=='line' and min(h,w)==1 and max(h,w)>=2 and len(s)==h*w:return True
        if fam=='L' and min(h,w)>=2:
            expected={(r,0) for r in range(h)}|{(h-1,c) for c in range(w)}
        elif fam=='T' and h>=2 and w>=3 and w%2:
            expected={(0,c) for c in range(w)}|{(r,w//2) for r in range(h)}
        elif fam=='U' and h>=2 and w>=3:
            expected={(r,c) for r in range(h) for c in [0,w-1]}|{(h-1,c) for c in range(w)}
        elif fam=='H' and min(h,w)>=3 and h%2:
            expected={(r,c) for r in range(h) for c in [0,w-1]}|{(h//2,c) for c in range(w)}
        elif fam=='plus' and min(h,w)>=3 and h%2 and w%2:
            expected={(h//2,c) for c in range(w)}|{(r,w//2) for r in range(h)}
        elif fam=='frame' and min(h,w)>=3:
            expected={(r,c) for r in range(h) for c in range(w) if r in [0,h-1] or c in [0,w-1]}
        else:continue
        if s==expected:return True
    return False

def solve59(grid,prompt):
    a=np.asarray(grid);oc,fam,template,rule=parse59(prompt);oc=color(oc,a);comps=components(a==oc)
    vs=variants(template,rule) if template else None
    flags=[norm(c) in vs if vs else generic59(c,fam,rule,prompt) for c in comps]
    matches=sum(flags)
    extra={'mode':'exact' if template else 'generic','family':fam,'rule':rule,'n_matches':matches,'n_target_components':len(comps),'n_distractors':len(comps)-matches,'n_target_cells':int((a==oc).sum()),'all_target_correct':all(flags),'target_occupancy':int((a==oc).sum())/a.size}
    if template:
        sizes=[len(c)==len(template) for c in comps];boxes=[dims(c) in {dims(v) for v in vs} for c in comps]
        extra.update(template_area=len(template),template_h=dims(template)[0],template_w=dims(template)[1],size_only_correct=sum(sizes)==matches,box_only_correct=sum(boxes)==matches,n_same_size_nearmisses=sum(q and not f for q,f in zip(sizes,flags)))
        for ru in ['exact','rotations','dihedral']:
            extra[ru+'_only_correct']=sum(norm(c) in variants(template,ru) for c in comps)==matches
    return str(matches),extra

def solve60(grid,prompt):
    a=np.asarray(grid);oc=color(re.search(r'line of color (\w+)',prompt).group(1),a);n=int(re.search(r'Return the last (\d+) cells? from',prompt).group(1))
    cells=set(map(tuple,np.argwhere(a==oc)))
    adj={p:[(p[0]+dr,p[1]+dc) for dr,dc in D4 if (p[0]+dr,p[1]+dc) in cells] for p in cells}
    ends=sorted(p for p,v in adj.items() if len(v)==1)
    if len(ends)!=2 or any(len(v) not in (1,2) for v in adj.values()):raise ValueError('not a simple path')
    ordered=[ends[0]];prior=None
    while len(ordered)<len(cells):
        ns=[p for p in adj[ordered[-1]] if p!=prior]
        if len(ns)!=1:raise ValueError('path disconnected')
        prior=ordered[-1];ordered.append(ns[0])
    if ordered[-1]!=ends[1] or len(set(ordered))!=len(cells) or n>len(cells):raise ValueError('bad path')
    def fmt(ps):return '['+','.join(f'({r},{c})' for r,c in ps)+']'
    out='first:'+fmt(ordered[:n])+', second:'+fmt(ordered[::-1][:n])
    directions=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(ordered,ordered[1:])]
    turns=sum(x!=y for x,y in zip(directions,directions[1:]));endturns=sum(x!=y for x,y in zip(directions[:n-1],directions[1:n-1]))+sum(x!=y for x,y in zip(directions[-(n-1):],directions[-(n-1):][1:])) if n>1 else 0
    h,w=a.shape
    return out,{'n':n,'path_length':len(cells),'turns':turns,'end_turns':endturns,'overlap':2*n>len(cells),'path_occupancy':len(cells)/a.size,'path_h':dims(cells)[0],'path_w':dims(cells)[1],'path_bbox_occupancy':dims(cells)[0]*dims(cells)[1]/a.size,'both_ends_on_border':all(r in (0,h-1) or c in (0,w-1) for r,c in ends),'some_end_on_border':any(r in (0,h-1) or c in (0,w-1) for r,c in ends),'row_major_correct':'first:'+fmt(sorted(cells)[:n])+', second:'+fmt(sorted(cells)[::-1][:n])==out}

SOLVERS={'56':solve56,'57':solve57,'58':solve58,'59':solve59,'60':solve60}
