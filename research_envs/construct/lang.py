"""The spec language: a whitelisted subset of Python expressions.

A spec expression is parsed with `ast`, checked node by node against a whitelist, and only then compiled. There is
no attribute access, no imports, no lambdas, no names outside an allowed set and no dunder names, so the compiled code
can reach nothing but the functions we hand it. Sizes that would run inside C code (where a timeout cannot interrupt)
are capped by wrapper functions: `range`, `combinations`, `product` and `pow` refuse arguments past fixed limits.
Evaluation runs under a wall-clock timeout (`timed`).
"""
import ast
import itertools
import signal
from contextlib import contextmanager


class SpecError(ValueError):
    """The expression is not in the language (raised at validation)."""


class EvalTimeout(RuntimeError):
    """An evaluation ran past its time limit."""


ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Call, ast.Name, ast.Load, ast.Store, ast.Constant, ast.Tuple, ast.List, ast.Set,
    ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.comprehension, ast.IfExp, ast.Subscript,
)
MAX_RANGE = 10 ** 6
MAX_POOL = 4096          # largest iterable combinations/product will draw from
MAX_CHOOSE = 6           # largest k in combinations
MAX_EXPONENT = 64


class StepBudget(RuntimeError):
    """An evaluation drew more items from the iteration helpers than its budget allows."""


STEPS = {'left': 0}
MAX_STEPS = 5 * 10 ** 6


def _metered(it):
    """Yield from `it` one item at a time in Python bytecode, so a timeout can interrupt it, charging the budget."""
    for x in it:
        STEPS['left'] -= 1
        if STEPS['left'] < 0:
            raise StepBudget(f'more than {MAX_STEPS} iteration steps')
        yield x


def reset_steps(n=MAX_STEPS):
    STEPS['left'] = n


def _range(*args):
    for a in args:
        if not isinstance(a, int) or isinstance(a, bool):
            raise SpecError('range takes integers')
    r = range(*args)
    if len(r) > MAX_RANGE:  # object-dependent sizes are a resource limit (the object is invalid), not a spec error
        raise StepBudget(f'range longer than {MAX_RANGE}')
    return _metered(r)


def _combinations(pool, k):
    pool = tuple(pool)
    if not isinstance(k, int) or k < 0 or k > MAX_CHOOSE:
        raise SpecError(f'combinations limited to k <= {MAX_CHOOSE}')
    if len(pool) > MAX_POOL:
        raise StepBudget(f'combinations over a pool of {len(pool)} items (limit {MAX_POOL})')
    return _metered(itertools.combinations(sorted(pool, key=_order), k))


def _product(*pools):
    pools = [tuple(p) for p in pools]
    if len(pools) > MAX_CHOOSE:
        raise SpecError(f'product of more than {MAX_CHOOSE} pools')
    if any(len(p) > MAX_POOL for p in pools):
        raise StepBudget(f'product over a pool larger than {MAX_POOL}')
    return _metered(itertools.product(*pools))


def _pow(a, b):
    if not isinstance(b, int) or abs(b) > MAX_EXPONENT or abs(a) > 10 ** 6:
        raise SpecError('pow limited to small integers')
    return a ** b


def _order(x):
    """A total order on the values the language produces (ints, tuples, frozensets), for deterministic iteration."""
    if isinstance(x, frozenset):
        return (2, tuple(sorted(x, key=_order)))
    if isinstance(x, tuple):
        return (1, tuple(_order(y) for y in x))
    return (0, x)


def _sorted(xs):
    return sorted(xs, key=_order)


BUILTINS = {
    'all': all, 'any': any, 'sum': sum, 'len': len, 'min': min, 'max': max, 'abs': abs,
    'range': _range, 'combinations': _combinations, 'product': _product, 'pow': _pow,
    'set': frozenset, 'frozenset': frozenset, 'tuple': tuple, 'sorted': _sorted,
    'True': True, 'False': False,
    '_numeric': lambda a, b: all(isinstance(x, int) and not isinstance(x, bool) for x in (a, b)),
}


class _Validator(ast.NodeVisitor):
    def __init__(self, names, private_ok=()):
        self.names = set(names)
        self.private_ok = set(private_ok)
        self.bound = []

    def generic_visit(self, node):
        if not isinstance(node, ALLOWED_NODES):
            raise SpecError(f'{type(node).__name__} is not allowed')
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith('_') and node.id not in self.private_ok:
            raise SpecError(f'name {node.id!r} is not allowed')
        if isinstance(node.ctx, ast.Load) and node.id not in self.names and not any(node.id in b for b in self.bound):
            raise SpecError(f'unknown name {node.id!r}')

    def visit_Constant(self, node):
        if not isinstance(node.value, (int, bool)) or isinstance(node.value, complex):
            raise SpecError(f'constant {node.value!r} is not allowed (integers and booleans only)')

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise SpecError('only named functions may be called')
        if node.keywords:
            raise SpecError('keyword arguments are not allowed')
        self.visit(node.func)
        for a in node.args:
            if isinstance(a, ast.Starred):
                raise SpecError('star arguments are not allowed')
            self.visit(a)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Pow):
            if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, int)
                    and abs(node.right.value) <= MAX_EXPONENT):
                raise SpecError(f'** needs a constant exponent of at most {MAX_EXPONENT}; use pow() otherwise')
        self.generic_visit(node)

    def _comp(self, node):
        targets = set()
        for gen in node.generators:
            if gen.is_async:
                raise SpecError('async comprehension')
            self.visit(gen.iter)
            for t in ast.walk(gen.target):
                if isinstance(t, ast.Name):
                    if t.id.startswith('_'):
                        raise SpecError(f'name {t.id!r} is not allowed')
                    targets.add(t.id)
                elif not isinstance(t, (ast.Tuple, ast.Store)):
                    raise SpecError('comprehension targets must be names or tuples of names')
            self.bound.append(targets)
            for cond in gen.ifs:
                self.visit(cond)
        elt = [node.elt]
        for e in elt:
            self.visit(e)
        for _ in node.generators:
            self.bound.pop()

    visit_GeneratorExp = visit_ListComp = visit_SetComp = _comp


def graded(src, names):
    """A graded violation measure for a boolean constraint, for search only (0 when satisfied):
    all(GEN) -> number of failing instances; a single numeric comparison -> distance from the bound;
    A and B -> sum of the parts; anything else -> 0 or 1."""
    tree = ast.parse(src.strip(), mode='eval')
    _Validator(set(names) | set(BUILTINS)).visit(tree)
    body = _grade(tree.body)
    expr = ast.Expression(body=body)
    ast.fix_missing_locations(expr)
    _Validator(set(names) | set(BUILTINS), private_ok={'_numeric'}).visit(expr)
    return compile(expr, '<graded>', 'eval')


def _call(name, *args):
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[])


def _indicator(node):
    return ast.IfExp(test=node, body=ast.Constant(0), orelse=ast.Constant(1))


def _grade(node):
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        out = _grade(node.values[0])
        for v in node.values[1:]:
            out = ast.BinOp(left=out, op=ast.Add(), right=_grade(v))
        return out
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'all'
            and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp)):
        # the sum of each instance's own grade (0 iff it holds): all(deg(v) == d ...) grades as the total distance of
        # the degrees from d, not the number of failing vertices, so every step toward d counts
        gen = node.args[0]
        gens = [ast.comprehension(target=g.target, iter=g.iter, ifs=list(g.ifs), is_async=0) for g in gen.generators]
        return _call('sum', ast.GeneratorExp(elt=_grade(gen.elt), generators=gens))
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.LtE, ast.Lt, ast.GtE, ast.Gt)):
        a, b, op = node.left, node.comparators[0], node.ops[0]
        diff = {ast.Eq: _call('abs', ast.BinOp(left=a, op=ast.Sub(), right=b)),
                ast.LtE: _call('max', ast.Constant(0), ast.BinOp(left=a, op=ast.Sub(), right=b)),
                ast.Lt: _call('max', ast.Constant(0), ast.BinOp(left=ast.BinOp(left=a, op=ast.Sub(), right=b), op=ast.Add(), right=ast.Constant(1))),
                ast.GtE: _call('max', ast.Constant(0), ast.BinOp(left=b, op=ast.Sub(), right=a)),
                ast.Gt: _call('max', ast.Constant(0), ast.BinOp(left=ast.BinOp(left=b, op=ast.Sub(), right=a), op=ast.Add(), right=ast.Constant(1)))}
        # only when both sides are numeric at run time; otherwise fall back to an indicator
        return ast.IfExp(test=_call('_numeric', a, b), body=diff[type(op)], orelse=_indicator(node))
    return _indicator(node)


def compile_expr(src, names):
    """Validate `src` against the language with the given extra names and return a code object."""
    if not isinstance(src, str) or len(src) > 4000:
        raise SpecError('expression must be a string of at most 4000 characters')
    try:
        tree = ast.parse(src.strip(), mode='eval')
    except SyntaxError as e:
        raise SpecError(f'syntax error: {e.msg}') from None
    _Validator(set(names) | set(BUILTINS)).visit(tree)
    return compile(tree, '<spec>', 'eval')


def run(code, env):
    """Evaluate compiled spec code in an environment of our own functions and values only."""
    reset_steps()
    scope = {'__builtins__': {}}  # names resolve as globals, so comprehensions see them too
    scope.update(BUILTINS)
    scope.update(env)
    return eval(code, scope)  # noqa: S307 - code was validated by _Validator


@contextmanager
def timed(seconds):
    """Raise EvalTimeout if the body runs longer than `seconds` (main thread only)."""
    def handler(signum, frame):
        raise EvalTimeout(f'evaluation exceeded {seconds} s')
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
