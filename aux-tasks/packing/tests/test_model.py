import pytest

from pv2.model import Answer, Part, Query, Slot, fill


def test_fill_placeholders():
    assert fill("Output row 3 of ⟪g0⟫.", ["the third grid"]) == "Output row 3 of the third grid."
    assert fill("⟪G0⟫ has ⟪answer⟫ rows.", ["grid B"], answer="7") == "Grid B has 7 rows."
    assert fill("⟪g0s⟫ first cell", ["the second grid"]) == "the second grid's first cell"
    with pytest.raises(ValueError):
        fill("⟪g1⟫", ["x"])
    with pytest.raises(ValueError):
        fill("⟪answer⟫", ["x"])


def test_answer_contract():
    Answer("number", 7, "7")
    Answer("grid", [[1]])
    with pytest.raises(ValueError):
        Answer("grid", [[1]], "1")
    with pytest.raises(ValueError):
        Answer("number", 7)
    with pytest.raises(ValueError):
        Answer("float", 1.0, "1.0")


def test_part_shape_checks():
    s = Slot([[1, 2]])
    q = Query("lib", "lib:dims", (0,), "How big is ⟪g0⟫?", Answer("text", "1x2", "1x2"))
    Part("gallery", [s], [q]).validate_shape()
    bad = Query("lib", "lib:dims", (1,), "How big is ⟪g0⟫?", Answer("text", "1x2", "1x2"))
    with pytest.raises(ValueError):
        Part("gallery", [s], [bad]).validate_shape()
    unnamed = Query("lib", "lib:dims", (0,), "How big is it?", Answer("text", "1x2", "1x2"))
    with pytest.raises(ValueError):
        Part("gallery", [s], [unnamed]).validate_shape()


def test_generator_package_is_never_shadowed_by_the_parent_workspace_copy(tmp_path):
    """Items verifier (2026-09-29): run from a folder holding an older registry.py / prompts.py, that
    copy shadowed the generator package, pv2 loaded old task ids and built rows silently wrong."""
    import os
    import subprocess
    import sys
    from pv2.paths import DEFAULT_GENERATOR_ROOT, REPO_ROOT
    (tmp_path / "registry.py").write_text("def list_tasks():\n    return [str(i) for i in range(1, 103)]\n")
    code = ("import registry, pv2.paths as p; root = p.ensure_generator_package(); "
            "print(root); print(len([t for t in registry.list_tasks()]))")
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PV2_GENERATOR_ROOT", None)
    # importing pv2 first, from a folder with an old registry.py (the shadowing case)
    out = subprocess.run([sys.executable, "-c", "import pv2.paths as p; p.ensure_generator_package(); " + code],
                         cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    root, n = out.stdout.strip().splitlines()[-2:]
    assert os.path.abspath(root) == os.path.abspath(DEFAULT_GENERATOR_ROOT)
    assert int(n) != 102
    # an old registry imported before pv2 is refused, not used
    bad = subprocess.run([sys.executable, "-c", "import registry; import pv2.paths as p; p.ensure_generator_package()"],
                         cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert bad.returncode != 0 and "different registry" in bad.stderr
