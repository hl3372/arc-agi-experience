"""Registration wrappers for binary logic and isolation tasks."""
from registry import register


@register("42")
def generate_binary_logic(rng):
    from generators.binary_logic_42 import generate
    return generate(rng)


@register("46a")
def generate_count_isolated(rng):
    from generators.isolation_46_48 import generate
    return generate(rng)


@register("47")
def generate_remove_isolated(rng):
    from generators.isolation_46_48 import generate
    return generate(rng, remove=True)
