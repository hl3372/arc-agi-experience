"""Locate the generator package (the folder above ``packing/``) and put it on sys.path.

The generator package is imported, never copied: ``registry.generate`` and ``prompts.get_prompt``
are the source of every item.  ``PV2_GENERATOR_ROOT`` overrides the default location (a training
tree that already has ``registry.py`` importable needs nothing).
"""
from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # packing/
MODULE_ROOT = os.path.dirname(REPO_ROOT)                                      # aux-tasks/ (the generators)
WORKSPACE_ROOT = MODULE_ROOT
DEFAULT_GENERATOR_ROOT = MODULE_ROOT

# Optional cross-checks against the solver's own prompt format: its grid renderer (arc_render.py,
# ``parse_grids``) and its system prompts (prompts/*.md).  In this repository they live in
# ../training/arcsolver/prompt; PV2_SOLVER_PROMPT_DIR overrides.  When absent, those checks are skipped.
SOLVER_PROMPT_DIR = os.path.abspath(os.environ.get(
    "PV2_SOLVER_PROMPT_DIR", os.path.join(os.path.dirname(MODULE_ROOT), "training", "arcsolver", "prompt")))


def _check_api() -> None:
    import prompt_variants
    import prompts
    import registry
    missing = [name for mod, name in ((registry, "generate"), (registry, "list_tasks"), (prompts, "get_prompt"),
                                      (prompt_variants, "variants_for"), (prompt_variants, "render"))
               if not hasattr(mod, name)]
    if missing:
        raise ImportError(f"the imported generator package ({os.path.dirname(registry.__file__)}) lacks "
                          f"{missing}: an old copy is shadowing task_gen_review_workspace; set PV2_GENERATOR_ROOT")


def ensure_generator_package() -> str:
    """Make ``import registry`` resolve to the generator package; return its directory.

    When the generator root exists (``PV2_GENERATOR_ROOT`` or the default beside this repo) it goes
    FIRST on sys.path, so an older ``registry.py`` / ``prompts.py`` in the working directory (the
    parent workspace has one) can never shadow it; a registry already imported from elsewhere is an
    error.  Without that root (an assembled training tree ships the package flat), the importable
    package is accepted after an API check.
    """
    root = os.path.abspath(os.environ.get("PV2_GENERATOR_ROOT", DEFAULT_GENERATOR_ROOT))
    if os.path.isfile(os.path.join(root, "registry.py")):
        if sys.path[:1] != [root]:
            if root in sys.path:
                sys.path.remove(root)
            sys.path.insert(0, root)
        loaded = sys.modules.get("registry")
        if loaded is not None and os.path.dirname(os.path.abspath(loaded.__file__)) != root:
            raise ImportError(f"a different registry is already imported ({loaded.__file__}); "
                              f"import pv2 before it, or set PV2_GENERATOR_ROOT")
        import registry
        if os.path.dirname(os.path.abspath(registry.__file__)) != root:
            raise ImportError(f"registry resolved to {registry.__file__}, not {root}")
        _check_api()
        return root
    try:
        import registry
    except ImportError as err:
        raise ImportError(f"generator package not found at {root!r}; set PV2_GENERATOR_ROOT") from err
    _check_api()
    return os.path.dirname(os.path.abspath(registry.__file__))
