"""A test answer copied from a training answer fails at realization, a re-planned task commits, tasks can be held back
from a commit, the summary
lists tasks to read before commit, and the grid measures read a resized output's computed cells off the input's ground."""
from types import SimpleNamespace
import json
from arcgen.batch import cli, commit, properties
from arcgen.batch import runtime as batch_runtime


def _grid(h, w, f):
    return [[f(r, c) for c in range(w)] for r in range(h)]


PICTURE = _grid(9, 9, lambda r, c: (r * r + 2 * c + 3 * r * c) % 5)  # 9x9, five colours, 51 cells off its ground


def _task(train_out, test_out):
    return {'train': [{'input': _grid(9, 9, lambda r, c: 0), 'output': train_out}],
            'test': [{'input': _grid(10, 10, lambda r, c: (r + c) % 2), 'output': test_out}]}


def test_a_test_answer_copied_from_a_training_answer_is_reused():
    turned = [list(r) for r in zip(*PICTURE[::-1])]  # a quarter turn
    renamed = [[(c + 1) % 5 for c in row] for row in turned]  # and every colour renamed
    window = [row[1:8] for row in renamed[1:9]]  # an 8x7 part of it
    assert batch_runtime.reused_tests(_task(PICTURE, window)) == [
        'test 1 answer is a part of training output 1, turned, reflected or recoloured']
    grown = [row + [9] for row in renamed]  # the training answer makes up most (81 of 90 cells) of a larger answer
    assert batch_runtime.reused_tests(_task(PICTURE, grown)) == [
        'training output 1, turned, reflected or recoloured, makes up most of test 1 answer']
    broken = [row[:] for row in window]; broken[3][3] = (broken[3][3] + 1) % 5  # one cell off breaks the renaming
    assert batch_runtime.reused_tests(_task(PICTURE, broken)) == []
    small = [row[:5] for row in PICTURE[:5]]  # fewer than 30 cells off its ground: exempt
    assert batch_runtime.reused_tests(_task(PICTURE, small)) == []
    v1 = SimpleNamespace(q2=SimpleNamespace(prompt_set='v1')); r23 = SimpleNamespace(q2=SimpleNamespace(prompt_set='r23'))
    assert batch_runtime.Runtime.episode_defect(v1, _task(PICTURE, window)).startswith('reused test: test 1 answer is a part')
    assert batch_runtime.Runtime.episode_defect(r23, _task(PICTURE, window)) == ''


def test_a_replanned_task_binds_its_planner_and_holds_are_read():
    plan = {'purpose': 'planning', 'parent_ids': ['t1'], 'model': 'm', 'reasoning_effort': 'xhigh', 'account': 'a1'}
    ledger = [{**plan, 'call_id': 't1-replan-concrete-plan'}, {**plan, 'call_id': 't1-concrete-plan'},
              {**plan, 'call_id': 't2-concrete-plan', 'parent_ids': ['t2']}]
    assert commit.planner_call(ledger, 't1')['call_id'] == 't1-concrete-plan'
    assert commit.planner_call(ledger[2:], 't2')['call_id'] == 't2-concrete-plan'
    import pytest
    with pytest.raises(ValueError):
        commit.planner_call([ledger[0], {**ledger[1], 'model': 'other'}], 't1')
    with pytest.raises(ValueError):
        commit.planner_call(ledger, 't3')


def test_hold_files_take_a_map_or_a_list(tmp_path):
    (tmp_path / 'a.json').write_text(json.dumps({'t1': 'human no'}))
    (tmp_path / 'b.json').write_text(json.dumps(['t1', {'task_id': 't2', 'reason': 'reused answer'}]))
    assert commit.read_holds(tmp_path / 'a.json') == {'t1': 'human no'}
    assert commit.read_holds(tmp_path / 'b.json') == {'t1': 'held by the owner', 't2': 'reused answer'}


def test_read_first_flags_blind_failures_and_long_inferred_rules(tmp_path):
    rows = []
    for tid, words, solved in (('t1', 20, True), ('t2', 90, True), ('t3', 10, False)):
        down = tmp_path / tid / 'downstream'; cand = tmp_path / tid / 'candidates' / tid
        down.mkdir(parents=True); cand.mkdir(parents=True)
        (cand / 'blind.r0.json').write_text(json.dumps([{'response': {'primary_inferred_rule': 'word ' * 5}}]))
        (cand / 'blind.r1.json').write_text(json.dumps([{'response': {'primary_inferred_rule': 'word ' * words}}]))
        rows.append({'task_id': tid, 'state': {'path': str(down / 'state.json')}, 'family': 'SIM', 'rung': 'R1',
                     'blind_performance': {'any_all_test_success': solved}})
    value = cli.read_first(rows)
    assert value['flagged'] == {'t2': 'long rule: the blind solver needed 90 words',
                                't3': 'blind failed: the blind solver missed every test'}
    assert value['words_median'] == {'SIM R1': 20}


def test_resized_outputs_count_cells_off_the_input_ground_and_repeated_outputs_are_counted():
    # pieces (colour 8) outnumber the input's ground (1, on its border); the assembled answer is all piece cells
    test_in = _grid(12, 12, lambda r, c: 1 if r in (0, 11) or c in (0, 11) else 8)
    test_out = _grid(8, 9, lambda r, c: 8)
    task = {'train': [{'input': test_in, 'output': test_out}], 'test': [{'input': test_in, 'output': test_out}]}
    assert properties.scale(task)['computed_cells'] == 72
    twice = {'train': [{'input': _grid(9, 9, lambda r, c: 0), 'output': PICTURE},
                       {'input': _grid(9, 9, lambda r, c: 1), 'output': [list(r)[::-1] for r in PICTURE]}], 'test': []}
    assert properties.repeated_outputs(twice) and not properties.repeated_outputs({'train': twice['train'][:1], 'test': []})
