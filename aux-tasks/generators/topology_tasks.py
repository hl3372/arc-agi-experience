"""Registry entry points for the clean topology tasks; task 22 lives in count_holes."""
from registry import register


@register("36")
def generate_open_closed(rng):
    from generators.clean_topology_36_37 import generate_open_closed as generate
    return generate(rng)


@register("37")
def generate_close_open_object(rng):
    from generators.clean_topology_36_37 import generate_close_open_object as generate
    return generate(rng)
