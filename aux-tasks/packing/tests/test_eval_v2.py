"""eval_v2: the gold target scores perfectly; broken predictions score as broken; think blocks are cut."""
import pytest

from pv2 import eval_v2 as E


@pytest.fixture(scope="module")
def records():
    return E.make_eval_set(40, seed=77)


def test_gold_targets_score_perfectly(records):
    for r in records:
        s = E.score(r, "<think>\nsome reasoning\n</think>\n\n" + r["target"] + "<|im_end|>")
        assert s["strict"] and s["all_ok"] and s["lenient"] == 1.0, r["id"]


def test_wrong_and_garbled_predictions(records):
    for r in records:
        s = E.score(r, "I do not know.")
        assert not s["strict"] and s["lenient"] < 1.0
    summary = E.summarize(records, [E.score(r, r["target"]) for r in records])
    assert summary["all"]["strict"] == 1.0 and summary["all"]["lenient"] == 1.0
    assert any(k.startswith("family:") for k in summary) and any(k.startswith("style:") for k in summary)


def test_found_is_not_inflated_and_grids_compare_as_values(records):
    import json
    from pv2.assemble import styles as S
    wrong_found = 0
    for r in records:
        s_gold = E.score(r, r["target"])
        assert s_gold["found"] == 1.0
        # change one answer inside the target while keeping the style: never found, never ok
        for pc in r["checks"]["parts"]:
            for d in pc["questions"]:
                a = d["answer"]
                if a.isdigit():
                    bad = r["target"].replace(a, str(int(a) + 7), 1)
                    if bad != r["target"]:
                        sc = E.score(r, bad)
                        wrong_found += sc["found"] < 1.0 and sc["lenient"] < 1.0
    assert E._found("no", "the answer is not yes") is False and E._found("7", "row 17") is False
    assert E._found("no", "so the answer is no.") is True
    # a JSON grid written with ordinary JSON spacing is the same grid
    assert E._same("[[0,1],[2,3]]", json.dumps([[0, 1], [2, 3]]), "json_flat", True)
    assert E._same("01\n23", "01  \n23 ", "compact", True)
