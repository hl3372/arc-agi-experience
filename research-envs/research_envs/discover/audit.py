"""Build and audit one domain A world: rivals, queries, identifiability, needles, public baselines, demand audit.

Everything here is mechanical. The planner supplies the world, named rivals and a demand-suited experiment plan;
the queries are sampled by code. Terms (DESIGN_A section 11):
- interface-equivalent rival: no experiment within the length limit separates it from the truth;
- critical query: some separable rival in R answers it differently from the truth;
- local identifiability: the R-aware eliminator, within budget, leaves no rival that answers a published query
  differently (R-relative; not evidence of discoverability);
- needle: a named or structural rival that the public random sampler separates in under 1/B of experiments and that
  neither the sweep nor random experiments at budget B separate;
- public learner: the cross-validation-selected learner on public data; oracle learner: the best learner on the
  hidden queries (used only to make shortcuts as strong as possible).
"""
import json
import os
import random
from . import baselines as Bl, fairness as F, rivals as Rv

CRIT_OK = 0.9          # a learner "reaches" the critical queries at this accuracy
GAP_PTS = 0.25         # a shortcut "falls short" by at least this much...
GAP_MIN = 5            # ...and by at least this many critical queries


def _sig(w, q):
    return F.answer(w, q)


def _demand_rival(label):
    return label.startswith('named') or 'structural' in label


TRIVIAL_CAP = float(os.environ['RE_A_TRIVIAL_CAP']) if os.environ.get('RE_A_TRIVIAL_CAP') else None  # demo round: max share of "nothing changes" queries
ANSWER_CAP = float(os.environ['RE_A_ANSWER_CAP']) if os.environ.get('RE_A_ANSWER_CAP') else None  # demo repair: max share of queries with one answer value


QUERY_MODE = os.environ.get('RE_A_QUERY_MODE')  # 'discriminating': every query refutes rivals (demo v2)


def choose_queries(truth, rivals, rng, n=30, n_crit=12, pool=400, extra_len=2, exclude=()):
    """Sample a pool, mark critical queries (those on which a named or structural rival, a misconception the task
    is about, disagrees with the truth), keep at least n_crit critical ones when there are enough, fill the rest at
    random. 'any_rival' records every disagreeing rival. Extrapolation queries are kept apart as a diagnostic."""
    cand = F.sample_queries(truth, rng, n_interp=pool * 2 // 3, n_comp=pool // 3, extra_len=0)
    starts = truth.starts()
    n_in, L = len(truth.inputs), truth.max_len
    for k in range(1, L):              # structured probes: a run of one action, then another (how careful
        for i in range(n_in):          # experimenters probe), from spread starts
            for j in range(n_in):
                for _ in range(2):
                    cand.append({'regime': 'structured', 'start': starts[rng.randrange(len(starts))],
                                 'inputs': [i] * k + [j]})
    for q in cand:
        a = _sig(truth, q)
        q['answer'] = a
        q['any_rival'] = [i for i, (_, w) in enumerate(rivals) if _sig(w, q) != a]
        q['disagree'] = [i for i in q['any_rival'] if _demand_rival(rivals[i][0])]
    for q in cand:
        s0 = truth.start_state(q['start'])
        q['trivial'] = list(q['answer']) == list(truth.out_tuple[s0])
    if QUERY_MODE == 'discriminating':
        chosen = _discriminating(cand, rivals, rng, n, set(exclude))
        return chosen, _extra(truth, rivals, rng, extra_len)
    crit = [q for q in cand if q['disagree']]
    rest = [q for q in cand if not q['disagree']]
    rng.shuffle(crit); rng.shuffle(rest)
    if TRIVIAL_CAP is not None:   # visible-change queries first; trivial ones only up to the cap
        crit.sort(key=lambda q: q['trivial'])
        rest.sort(key=lambda q: q['trivial'])
    # stratified by facet: round-robin over the demand rivals, so each misconception gets its own critical queries
    # (otherwise the rival that is wrong most often fills the set, and the subtle ones are never tested)
    by_rival = {}
    for q in crit:
        for i in q['disagree']:
            by_rival.setdefault(i, []).append(q)
    chosen, ids = [], set()
    by_answer = {}                     # answer diversity: no single answer value fills the query set

    def over_cap(q):
        return ANSWER_CAP is not None and by_answer.get(json.dumps(q['answer']), 0) >= max(1, int(ANSWER_CAP * n))

    def take(q):
        chosen.append(q); ids.add(id(q))
        by_answer[json.dumps(q['answer'])] = by_answer.get(json.dumps(q['answer']), 0) + 1
    while len(chosen) < n_crit and any(by_rival.values()):
        for i in sorted(by_rival, key=lambda i: len(by_rival[i])):
            while by_rival[i] and (id(by_rival[i][0]) in ids or over_cap(by_rival[i][0])):
                by_rival[i].pop(0)
            if by_rival[i] and len(chosen) < n_crit:
                take(by_rival[i].pop(0))
    if TRIVIAL_CAP is not None:
        cap = int(TRIVIAL_CAP * n)
        triv = sum(q['trivial'] for q in chosen)
        for q in rest + [q for q in crit if id(q) not in ids]:
            if len(chosen) >= n:
                break
            if id(q) in ids or (q['trivial'] and triv >= cap) or over_cap(q):
                continue
            take(q); triv += q['trivial']
    else:
        chosen += rest[:n - len(chosen)]
        if len(chosen) < n:
            chosen += [q for q in crit if id(q) not in ids][:n - len(chosen)]
    return chosen, _extra(truth, rivals, rng, extra_len)


def _extra(truth, rivals, rng, extra_len):
    extra = [q for q in F.sample_queries(truth, rng, n_interp=20, n_comp=0, extra_len=extra_len)
             if q['regime'] == 'extrapolation'] if extra_len else []
    for q in extra:
        q['answer'] = _sig(truth, q)
        q['any_rival'] = [i for i, (_, w) in enumerate(rivals) if _sig(w, q) != q['answer']]
        q['disagree'] = [i for i in q['any_rival'] if _demand_rival(rivals[i][0])]
    return extra


def _discriminating(cand, rivals, rng, n, exclude):
    """Every query earns its place: greedy choice of n queries so that, together, they refute every identifiable
    rival many times over. A query's gain is the sum over the rivals it refutes of w_r / (1 + times r is already
    refuted), with misconception rivals (named, structural) weighted 3x, so refutations spread across all rivals
    instead of piling on the easy ones, and queries that refute many rivals at once are preferred. Queries on which
    an excluded rival (not identifiable within budget, or a needle) disagrees are never chosen, so fairness holds by
    construction. The answer-value and "nothing changes" caps apply as before."""
    weight = [3.0 if _demand_rival(l) else 1.0 for l, _ in rivals]
    pool = [q for q in cand if q['any_rival'] and not (set(q['any_rival']) & exclude)]
    rng.shuffle(pool)
    count = [0] * len(rivals)
    acap = max(1, int(ANSWER_CAP * n)) if ANSWER_CAP is not None else n
    tcap = int(TRIVIAL_CAP * n) if TRIVIAL_CAP is not None else n
    chosen, taken, by_answer, triv = [], set(), {}, 0
    while len(chosen) < n:
        best, best_gain = None, 0.0
        for k, q in enumerate(pool):
            if k in taken or by_answer.get(json.dumps(q['answer']), 0) >= acap or (q['trivial'] and triv >= tcap):
                continue
            gain = sum(weight[i] / (1 + count[i]) for i in q['any_rival'])
            if gain > best_gain:
                best, best_gain = k, gain
        if best is None:
            break
        q = pool[best]
        taken.add(best); chosen.append(q)
        by_answer[json.dumps(q['answer'])] = by_answer.get(json.dumps(q['answer']), 0) + 1
        triv += q['trivial']
        for i in q['any_rival']:
            count[i] += 1
    return chosen


def learner_scores(truth, data, queries):
    crit = [q for q in queries if q['disagree']]
    out = {}
    for name, f in Bl.LEARNERS.items():
        m = f(data)
        out[name] = {'all': Bl.accuracy(m, truth, queries)[0], 'critical': Bl.accuracy(m, truth, crit)[0] if crit else None}
    name, m, cv = Bl.cv_select(data)
    out['public'] = {'learner': name, 'all': Bl.accuracy(m, truth, queries)[0],
                     'critical': Bl.accuracy(m, truth, crit)[0] if crit else None}
    best = max((k for k in out if k != 'public'), key=lambda k: (out[k]['critical'] or 0, out[k]['all']))
    out['oracle'] = {'learner': best, **out[best]}
    return out


def build(spec, demand, plan_experiments, named=(), budget=40, seed=0):
    """named: [{'label': ..., 'world': spec}] rival laws the planner says the task is designed around."""
    rng = random.Random(seed)
    truth, R = Rv.rival_set(spec, named)
    a = {'states': truth.n_states, 'inputs': len(truth.inputs), 'starts': len(truth.starts()), 'rivals': len(R)}
    # interface equivalence and needles
    sep = [F.first_difference(truth, w) for _, w in R]
    equiv = [i for i, d in enumerate(sep) if d is None]
    a['interface_equivalent'] = [R[i][0] for i in equiv]
    sw = Bl.sweep(truth, budget, random.Random(seed))
    rx = Bl.random_experiments(truth, budget, random.Random(seed + 1))
    ad = Bl.adaptive(truth, budget, random.Random(seed + 4))

    def found_by(exps, w):
        return any(F.trace(truth, e) != F.trace(w, e) for e in exps)
    needles = []
    for i, (label, w) in enumerate(R):
        if i in equiv or not (label.startswith('named') or 'structural' in label):
            continue
        frac = F.needle_fraction(truth, w, random.Random(seed + 2), n=300)
        if frac < 1 / budget and not any(found_by(x, w) for x in (sw, rx, ad)):
            needles.append({'rival': label, 'fraction': frac, 'separator': sep[i]})
    a['needles'] = needles
    # queries (interface-equivalent rivals never decide a query)
    separable = [(l, w) for i, (l, w) in enumerate(R) if i not in equiv]
    # local identifiability (R-relative); independent of the queries, so it is computed first and the
    # discriminating chooser can leave out queries that only an unidentifiable or needle rival would get wrong
    used, alive = F.eliminate(truth, [w for _, w in separable], budget, random.Random(seed + 3))
    needle_idx = [i for i, (l, _) in enumerate(separable) if any(n['rival'] == l for n in needles)]
    queries, extra = choose_queries(truth, separable, rng, exclude=set(alive) | set(needle_idx))
    a['queries'] = {'count': len(queries), 'critical': sum(bool(q['disagree']) for q in queries),
                    'extrapolation_diagnostic': len(extra), 'mode': QUERY_MODE or 'critical+fill'}
    undecided = [q for q in queries if any(i in q['any_rival'] for i in alive)]
    unfair_q = [q for q in queries if set(q['any_rival']) & (set(alive) | set(needle_idx))]
    a['identifiability'] = {'experiments_used': len(used), 'rivals_left': [separable[i][0] for i in alive],
                            'rivals_left_idx': list(alive), 'needle_idx': needle_idx,
                            'queries_undecided': len(undecided), 'queries_unfair': len(unfair_q)}
    # public baselines at budget B and 4B
    plan = [(e['start'], [truth.input_index(*x) if isinstance(x, list) else x for x in e['inputs']])
            if isinstance(e, dict) else e for e in plan_experiments][:budget]
    a['baselines'] = {}
    for label, exps in (('sweep', sw), ('random', rx), ('plan', plan), ('adaptive', ad),
                        ('sweep_4x', Bl.sweep(truth, 4 * budget, random.Random(seed))),
                        ('random_4x', Bl.random_experiments(truth, 4 * budget, random.Random(seed + 1)))):
        a['baselines'][label] = learner_scores(truth, Bl.collect(truth, exps), queries) if exps else None
    # coverage: can a full sweep of starts x all sequences up to L fit in 10x budget?
    space = a['starts'] * sum(len(truth.inputs) ** k for k in range(1, truth.max_len + 1))
    a['coverage'] = space <= 10 * budget
    a['audit'] = demand_audit(a, demand)
    return truth, R, queries, extra, a


def demand_audit(a, demand):
    """Two-sided, at equal budget: the demand-suited plan lets the public learner reach the critical queries, while
    the strongest shortcut (oracle learner on sweep or random data, at B and at 4B) falls short."""
    n_crit = a['queries']['critical']
    b = a['baselines']
    plan_only = (b.get('plan') or {}).get('public', {}).get('critical')
    adapt = (b.get('adaptive') or {}).get('public', {}).get('critical')
    plan = max((v for v in (plan_only, adapt) if v is not None), default=None)
    shortcut = {k: (b.get(k) or {}).get('oracle', {}).get('critical') for k in ('sweep', 'random', 'sweep_4x', 'random_4x')}
    best_short = max((v for v in shortcut.values() if v is not None), default=None)
    out = {'critical_queries': n_crit, 'plan_public': plan_only, 'adaptive_public': adapt, 'shortcut_oracle': shortcut}
    reasons = []
    if n_crit < GAP_MIN:
        reasons.append(f'only {n_crit} critical queries')
    if a['coverage']:
        reasons.append('coverage: a full sweep fits in 10x budget')
    # needles and undecided queries are handled per query (the pipeline drops the queries they affect); noted here
    out['notes'] = []
    if a['needles']:
        out['notes'].append(f"needle rivals (their queries dropped): {[n['rival'] for n in a['needles']]}")
    if a['identifiability']['queries_undecided']:
        out['notes'].append(f"{a['identifiability']['queries_undecided']} queries not identifiable within budget (dropped)")
    reached = plan is not None and plan >= CRIT_OK
    short = best_short is not None and (plan or 0) - best_short >= GAP_PTS and (plan or 0) * n_crit - best_short * n_crit >= GAP_MIN
    out.update({'plan_reaches': reached, 'shortcut_falls_short': short})
    if demand:
        out['passes'] = reached and short and not reasons
    else:
        out['passes'] = None
        out['best_shortcut_gap'] = None if best_short is None else 1 - best_short
    out['reasons'] = reasons
    return out
