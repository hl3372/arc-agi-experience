"""One explicit candidate contract selection; legacy runs have no extra defaults."""
from .factory_policy import AUTHORING_CONTRACT_PROFILE, pipeline_policy


def load_contract_bindings(config):
    if "contract_profile" not in config:
        return None
    if config["contract_profile"] != AUTHORING_CONTRACT_PROFILE or pipeline_policy(config) is None:
        raise ValueError("unsupported candidate contract/profile combination")
    try:
        from .astra_contract_bindings_v3 import build_contract_bindings
    except ImportError as exc:
        raise ValueError("selected candidate contract files are not installed; finalized contract packet required") from exc
    return build_contract_bindings()


def is_canonical_plan(plan):
    from collections.abc import Mapping
    return isinstance(plan, Mapping) and plan.get("contract_version") == "astra-authoring-v3"


def plan_fidelity(plan, realization):
    """Retain legacy checks; canonical records use the qualified structural view."""
    if is_canonical_plan(plan):
        from .astra_authoring_contract_v3 import structural_fidelity
        return structural_fidelity(plan, realization)
    from .v4_diagnostics import check_plan_fidelity
    return check_plan_fidelity(plan, realization)


def display_value(value, plan):
    """Resolve references only in a derived display, preserving canonical bytes."""
    if is_canonical_plan(plan):
        from .astra_authoring_contract_v3 import resolve_color_references
        return resolve_color_references(value, plan)
    return value
