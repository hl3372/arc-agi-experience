"""Synthetic fixtures for the offline suite: an episode (plan + realization), checker and solver responses, an idea
window, a selector response and fake embedding vectors. None of this is quality evidence; it exercises contracts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

from arcgen.lib import astra_authoring_schemas_v3 as s
from arcgen.lib.astra_authoring_contract_v3 import validate_realization
from arcgen.lib.util import content_hash

HERE = Path(__file__).resolve().parent
EPISODE = HERE / 'symbolic_episode.json'
WINDOW = HERE / 'screen_window.json'


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return content_hash(value) if not isinstance(value, str) else content_hash({'text': value})


def idea(candidate_id: str, mode: str, relation: str | None = None) -> dict:
    return {
        "candidate_id": candidate_id,
        "search_mode": mode,
        "central_relation": relation or f"Relation for {candidate_id}",
        "why_coherent": "One coherent mapping.",
        "novel_concept_potential": "A fresh inference.",
        "functional_composition_potential": "Several necessary signals interact.",
        "why_multiple_training_examples_may_be_useful": "They vary incidental evidence.",
        "possible_test_generalization": "The test applies the same abstraction.",
        "main_authoring_risk": "Avoid a local shortcut.",
    }


def selector_assessment(candidate_id: str, selected: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "novel_concept_potential": "solid",
        "functional_composition_potential": "solid",
        "fair_episode_potential": "solid",
        "derivative_of": "none",
        "duplicate_or_sibling_of": [],
        "overused_metaphor_or_attractor": "",
        "decision": "select" if selected else "reject",
        "reason": "Strongest candidate." if selected else "Lower development priority.",
        "similarity_scope": "none",
    }


def editorial_brief(candidate_id: str, priority: int = 1) -> dict:
    return {
        "candidate_id": candidate_id,
        "development_priority": priority,
        "route": "flagship",
        "preserve": "Preserve the relation.",
        "develop": "Make the evidence concrete.",
        "avoid": "Avoid shortcuts.",
        "closest_similarity_concern": "No close procedure.",
        "selection_reason": "Best balance of both axes.",
        "optional_development_suggestion": None,
    }


def concrete_plan(candidate_id: str) -> dict:
    pair = lambda n, purpose: {
        "pair_number": n,
        "pair_purpose": purpose,
        "input_dimensions": {"height": 3, "width": 3},
        "output_dimensions": {"height": 3, "width": 3},
        "exact_input_construction": f"Place one blue cell at the specified training-{n} location.",
        "exact_output_construction": "Change that cell to red in the same location.",
        "precise_placements_quantities_relations_or_states": "All unspecified cells are background zero.",
        "output_derivation": "Replace the sole 1 by 2.",
        "what_it_helps_communicate": "Position is incidental and color role is invariant.",
        "deliberate_overlap_with_other_pairs": "Both demonstrate the same role mapping.",
        "why_overlap_is_harmless": "The second varies position and confirms position invariance.",
    }
    test = {
        "pair_number": 1,
        "input_dimensions": {"height": 3, "width": 3},
        "output_dimensions": {"height": 3, "width": 3},
        "exact_input_construction": "Place one blue cell at the upper-right edge.",
        "exact_expected_output_construction": "Change the blue cell to red in place.",
        "precise_placements_quantities_relations_or_states": "Blue at row 0 column 2.",
        "output_derivation": "Replace 1 by 2.",
        "what_is_new_relative_to_training": "A new position only.",
        "training_support": "Both training pairs establish position invariance.",
        "test_only_semantics_audit": "No new role, condition, or operation.",
        "why_one_output_is_best_supported": "The invariant color-role mapping fixes it.",
    }
    return {
        "candidate_id": candidate_id,
        "status": "ready",
        "abort_reason": "",
        "semantic_core": {
            "central_rule": "Replace every blue signal cell with red without moving it.",
            "central_conceptual_insight": "The role, not position, determines the output.",
            "why_worth_realizing": "It is an exact mock episode.",
            "what_must_remain_visible_after_rendering": "Position varies while the role conversion is invariant.",
        },
        "episode_overview": {
            "training_pair_count": 2,
            "test_pair_count": 1,
            "intended_difficulty_and_inferability": "Simple and determinate.",
            "background_treatment": "Zero is empty canvas.",
            "background_color": 0,
            "functional_color_roles": [
                {"role": "signal", "color": 1, "functional_meaning": "cell to convert"},
                {"role": "result", "color": 2, "functional_meaning": "converted cell"},
            ],
        },
        "training_pairs": [pair(1, "establish"), pair(2, "reinforce")],
        "test_pairs": [test],
        "cross_pair_concrete_audit": {
            "output_relevant_properties": ["signal color"],
            "incidental_properties": ["position"],
            "relevant_and_incidental_variation": "Position changes across all pairs.",
            "strongest_simple_shortcuts": [
                {
                    "rule": "Change only the demonstrated positions.",
                    "fits_every_training_pair": True,
                    "comparably_simple": False,
                    "predicts_a_different_test_output": True,
                    "evidence_that_defeats_it": "The two distinct training positions establish a role mapping.",
                }
            ],
            "remaining_ambiguity_risk": "Low for the deterministic fixture.",
        },
        "canvas_plan": [
            {"pair": "train_1", "why_dimensions_are_needed": "Fixture", "functionally_used_areas": "One cell", "empty_space_role": "Shows location", "smaller_grid_audit": "Would remove variation", "larger_grid_audit": "No benefit"},
            {"pair": "train_2", "why_dimensions_are_needed": "Fixture", "functionally_used_areas": "One cell", "empty_space_role": "Shows location", "smaller_grid_audit": "Would remove variation", "larger_grid_audit": "No benefit"},
            {"pair": "test_1", "why_dimensions_are_needed": "Fixture", "functionally_used_areas": "One cell", "empty_space_role": "Shows location", "smaller_grid_audit": "Would remove variation", "larger_grid_audit": "No benefit"},
        ],
        "frozen_decisions": ["Two train pairs", "One test", "All dimensions 3 by 3", "Color mapping 1 to 2"],
        "allowed_realizer_freedom": ["No semantic freedom; only serialize the stated arrays"],
        "deterministic_realization_procedure": "Create zero 3x3 arrays, set the listed input location to 1 and corresponding output location to 2.",
        "axis_self_assessment": {"novel_concept": "solid", "functional_composition": "solid"},
    }


def realization(candidate_id: str) -> dict:
    zero = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    def grid(row: int, column: int, color: int) -> list[list[int]]:
        value = deepcopy(zero)
        value[row][column] = color
        return value
    deviations: list[dict] = []
    return {
        "candidate_id": candidate_id,
        "status": "realized",
        "refusal_reason": "",
        "task": {
            "train": [
                {"input": grid(1, 1, 1), "output": grid(1, 1, 2)},
                {"input": grid(2, 0, 1), "output": grid(2, 0, 2)},
            ],
            "test": [{"input": grid(0, 2, 1)}],
        },
        "authoring_metadata": {
            "intended_rule": "Replace the sole 1 by 2 in place.",
            "expected_test_outputs": [grid(0, 2, 2)],
            "construction_witness": "Explicit 3x3 coordinate construction.",
            "pair_construction_trace": [
                {"pair": "train_1", "trace": "(1,1): 1 becomes 2"},
                {"pair": "train_2", "trace": "(2,0): 1 becomes 2"},
                {"pair": "test_1", "trace": "(0,2): 1 becomes 2"},
            ],
            "plan_fidelity": {
                "statement": "The frozen plan is preserved exactly.",
                "preserved_frozen_decisions": ["Two train pairs", "One test", "All dimensions 3 by 3", "Color mapping 1 to 2"],
                "allowed_freedoms_used": ["serialized arrays"],
                "deviations": deviations,
            },
            "concrete_shortcut_audit": "No shorter different-output rule is comparably supported.",
            "canvas_use_classification": "compact_and_appropriate",
        },
        "deviations": deviations,
    }


def technical(candidate_id: str) -> dict:
    output = realization(candidate_id)["authoring_metadata"]["expected_test_outputs"]
    return {
        "task_id": candidate_id,
        "verifier_id": "technical-technical-primary",
        "status": "pass",
        "derived_test_outputs": output,
        "training_pair_traces": [
            {"pair": "train_1", "follows_rule_exactly": True, "trace": "1 becomes 2", "issues": []},
            {"pair": "train_2", "follows_rule_exactly": True, "trace": "1 becomes 2", "issues": []},
        ],
        "mechanically_correct": True,
        "test_supported": True,
        "one_answer_supported": True,
        "plan_fidelity": "pass",
        "unapproved_realizer_decisions": [],
        "evidence_sufficiency": "sufficient",
        "redundancy_assessment": "The confirmatory pair varies position usefully.",
        "pair_economy": "pass",
        "composition_preserved": True,
        "serious_concrete_shortcuts": [],
        "canvas_use": "appropriate",
        "arbitrary_exceptions": [],
        "unsupported_test_semantics": [],
        "diagnostic_flag_assessments": [],
        "repair_instruction": "",
        "confidence": "high",
    }


def blind(candidate_id: str, run_id: str) -> dict:
    output = realization(candidate_id)["authoring_metadata"]["expected_test_outputs"]
    return {
        "task_id": "blind-" + content_hash({"run": run_id, "candidate": candidate_id})[:12],
        "solver_id": "blind-blind-primary",
        "primary_inferred_rule": "Change 1 to 2 in place.",
        "predicted_test_outputs": output,
        "strongest_plausible_alternatives": [],
        "serious_different_output_alternative": False,
        "test_requires_unsupported_guessing": False,
        "fairness_assessment": "The answer is sufficiently supported.",
        "confidence": "high",
    }


def board(board_id: str, group_id: str, board_number: int, count: int = 7) -> dict:
    return {
        "board_id": board_id,
        "judgments": [
            {
                "task_id": f"{group_id}-b{board_number}-{index:02d}",
                "novel_concept": "standout",
                "functional_composition": "solid",
                "overall_quality": "real_quality",
                "discoverability": "The grids communicate the rule.",
                "conceptual_compression": "One compact relation.",
                "pair_sufficiency": "Enough evidence without overteaching.",
                "composition_grid_evidence": "Every role affects the output.",
                "scale_and_canvas_observation": "Compact and appropriate.",
                "same_minimal_procedure": False,
                "theme_is_common": False,
                "addition_is_excessive": False,
                "intrinsic_recommendation": "core_quality",
                "portfolio_recommendation": "include_now",
                "reason": "Clears the two-axis real-task bar.",
            }
            for index in range(1, count + 1)
        ],
        "batch_observations": "A valid deterministic fixture panel.",
    }



def episode(tid, train=2, test=1, no_ground=False):
    """A valid v3 plan and realization for one candidate: a marked cell moves two columns past a fixed anchor."""
    d = read(EPISODE); p, r = d['plan'], d['realization']; p['candidate_id'] = r['candidate_id'] = tid
    p['private_execution']['axis_self_assessment']['functional_composition'] = 'thin'
    pub, priv = p['public_contract'], p['private_execution']
    pub['episode_overview'].update(training_pair_count=train, test_pair_count=test)
    def resize(rows, n):
        result = [deepcopy(rows[i % len(rows)]) for i in range(n)]
        for i, x in enumerate(result, 1): x['pair_number'] = i
        return result
    pub['training_pairs'] = resize(pub['training_pairs'], train)
    pub['test_inputs'] = resize(pub['test_inputs'], test); priv['test_outputs'] = resize(priv['test_outputs'], test)
    for o in priv['test_outputs']: o['precise_placements_quantities_relations_or_states'] = 'ANSWER_CANARY_727893b1'
    template = deepcopy(priv['canvas_plan'][0])
    priv['canvas_plan'] = [dict(template, pair=f'{phase}_{i}') for phase, n in [('train', train), ('test', test)] for i in range(1, n + 1)]
    r['task']['train'] = [deepcopy(r['task']['train'][i % 2]) for i in range(train)]
    r['task']['test'] = [deepcopy(r['task']['test'][0]) for _ in range(test)]
    r['authoring_metadata']['expected_test_outputs'] = [deepcopy(r['authoring_metadata']['expected_test_outputs'][0]) for _ in range(test)]
    r['authoring_metadata']['pair_construction_trace'] = [{'pair': x['pair'], 'trace': 'Applied the fixed displacement.'} for x in priv['canvas_plan']]
    def grids(row, col):
        inp = [[0] * 8 for _ in range(4)]; out = deepcopy(inp)
        inp[3][7] = out[3][7] = 8; inp[row][col] = 1; out[row][col + 2] = 1
        return inp, out
    for i, pair in enumerate(r['task']['train']):
        inp, out = grids(i // 4, i % 4); pair.update(input=inp, output=out)
    for i, pair in enumerate(r['task']['test']):
        inp, out = grids(2 + i // 2, i % 2 + 2); pair['input'] = inp
        r['authoring_metadata']['expected_test_outputs'][i] = out
    if no_ground: pub['episode_overview']['palette_contract'].update(background_role=None, opt_out_reason='No single semantic ground.')
    validate_realization(p, r); return p, r


def source_window():
    """An idea window of twelve calls (four kernels each), its source rows and a selector response ranking them all."""
    w = read(WINDOW); source = {}; ass = []; rank = []
    for c in w['calls']:
        for k in c['opaque_kernel_ids']:
            source[k] = {'candidate_id': k, 'kernel': idea(k, 'open'),
                'private_source': {'J': 'episode_joint_inference', 'T': 'logical_leap', 'C': 'ordinary_bank_conditioned'}.get(c['arm'], 'ordinary_unconditioned'),
                'private_conditioned': c['arm'] == 'C', 'private_source_call_id': c['opaque_id'], 'private_effort': c['effort'],
                'study': {'primary_arm': c['arm'], 'invitation': c['invitation'], 'recipe_id': 'fixture-' + c['arm'],
                    'private_canary': 'SOURCE_CANARY_513e9c72', 'context': w['context'], 'window': w['id'], 'account': w['account']}}
            ass.append({'candidate_id': k, 'eligibility': 'plausible', 'conceptual_potential': 'solid', 'coherence_potential': 'solid',
                'fair_realization_potential': 'plausible', 'same_minimal_procedure_with': [], 'reason': 'Plausible coherent task.'})
            rank.append({'candidate_id': k, 'development_priority': len(rank) + 1,
                         'development_brief': 'Preserve the relation and support its application.', 'optional_development_suggestion': None})
    return w, source, {'selector_id': 'fixture', 'candidate_assessments': ass, 'ranked_plausible': rank, 'window_observations': None}


def fixture():
    """source_window() with the selector response fields of the current selector schema."""
    window, source, value = source_window()
    window['id'] = 'new-revision-fixture'
    for row in value['candidate_assessments']:
        row.pop('same_minimal_procedure_with')
        row.update(redundant_with=[], same_procedure_family_with=[], bank_comparisons=[],
                   useful_contribution='A concrete distinctive contribution from ' + row['candidate_id'])
    return window, source, value


def restrict(value, keys):
    """Keep only the given candidate IDs eligible, ranked in that order."""
    for row in value['candidate_assessments']:
        if row['candidate_id'] not in keys: row['eligibility'] = 'ineligible'
    indexed = {r['candidate_id']: r for r in value['ranked_plausible']}
    value['ranked_plausible'] = [dict(indexed[k], development_priority=i + 1) for i, k in enumerate(keys)]


def bank_rows(n=400):
    return [{'accepted_concept_id': 'a' + str(i), 'title': 'Title ' + str(i), 'concept_summary': 'Concept ' + str(i),
             'rule_summary': 'Rule ' + str(i)} for i in range(n)]


class Vectors:
    """Deterministic two-dimensional embeddings (no network)."""
    def __init__(self): self.calls = []

    def embed(self, texts):
        from arcgen.batch.common import digest as text_digest
        self.calls.append(list(texts))
        return [[1.0, int(text_digest(t)[:6], 16) / 0xffffff] for t in texts]


def references(n=45):
    """A synthetic bank reference corpus (accepted tasks only)."""
    return [{'reference_id': 'committed:' + str(i), 'source': 'accepted_generated_task', 'archive_tier': 'fixture',
             'central_relation': 'Relation ' + str(i), 'summary': 'Full procedural detail ' + str(i)} for i in range(n)]
