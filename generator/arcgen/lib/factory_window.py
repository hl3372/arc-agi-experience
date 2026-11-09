"""Maintained single-window selection, using the calibrated strict contract."""
from jsonschema import Draft202012Validator
from .factory_supply import SELECTOR_SCHEMAS
from .schemas import parse_json_object
from .v4_schemas import _transport_schema
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .factory_supply import FactorySupply, SPECIAL_SOURCES, DEFERRED, apply_source_ceilings
from .util import content_hash, load_prompt, read_json, render_prompt
from . import e18_supply as legacy


WINDOW_STAGE = "factory_window_selector"
WINDOW_SCHEMA = deepcopy(SELECTOR_SCHEMAS["v4_global_selector"])
WINDOW_SCHEMA["properties"]["candidate_assessments"]["maxItems"] = 48


def parse_window(data):
    value = parse_json_object(data) if isinstance(data, (str, bytes, bytearray)) else deepcopy(data)
    Draft202012Validator(WINDOW_SCHEMA).validate(value)
    assessed = [row["candidate_id"] for row in value["candidate_assessments"]]
    selected = [row["candidate_id"] for row in value["selected"]]
    decisions = {row["candidate_id"] for row in value["candidate_assessments"] if row["decision"] == "select"}
    if len(assessed) != len(set(assessed)) or len(selected) != len(set(selected)) or set(selected) != decisions:
        raise ValueError("selector membership/decisions inconsistent")
    if sorted(row["development_priority"] for row in value["selected"]) != list(range(1, len(selected) + 1)):
        raise ValueError("selector priorities must be contiguous")
    if sum(row.get("optional_development_suggestion") is not None for row in value["selected"]) > 2:
        raise ValueError("selector optional suggestion cap exceeded")
    return value


def window_bindings():
    from .factory_runtime import StageBinding
    return {WINDOW_STAGE: StageBinding(_transport_schema(WINDOW_SCHEMA), parse_window, ("full_window_selection",))}


def frontier(supply, response, slate, selection):
    """Apply ceilings to qualitative ranks, then develop up to twelve survivors.

    Keep up to four further eligible survivors as undeveloped overflow. Original
    selector ranks are preserved: raw ranks 13–16 may enter development after
    earlier exclusions. Never extend beyond the selector's qualified frontier.
    """
    private = {r['candidate_id']: r for r in supply['kernels']}
    assessments = {r['candidate_id']: r for r in response['candidate_assessments']}
    rows = []
    for brief in sorted(response['selected'], key=lambda r: int(r['development_priority'])):
        key = brief['candidate_id']
        row, assessment = private[key], assessments[key]
        rows.append({'source_candidate_id': key, 'lineage_id': content_hash({'e18_source': key}),
            'slate_index': slate, 'source_development_priority': int(brief['development_priority']),
            'search_mode': row['kernel']['search_mode'], 'kernel': deepcopy(row['kernel']),
            'editorial_brief': deepcopy(brief), 'private_source': row['private_source'],
            'private_conditioned': row['private_conditioned'], 'private_source_call_id': row['private_source_call_id'],
            'private_effort': row['private_effort'], 'selector_assessment': deepcopy(assessment),
            'novelty': {'selector_archive_checked': True, 'confirmed_same_procedure_duplicate': False,
                'within_window_sibling_ids': list(assessment.get('same_minimal_procedure_with') or []),
                'selected_as_sibling_representative': bool(assessment.get('same_minimal_procedure_with'))}})
    eligible, dropped = apply_source_ceilings(rows, selection)
    routine, routine_drops = apply_source_ceilings(eligible[:12], selection)
    return routine, eligible[12:16], dropped + routine_drops


class FullWindowSupply(FactorySupply):
    def select_slate(self, supply, slate):
        design = self.freeze_supply_design()
        directory = self.root / 'selection_slates' / f'slate-{slate:02d}'
        checkpoint = directory / 'selection_checkpoint.private.json'
        identity = content_hash({'factory': design['factory_identity'], 'supply': supply})
        if checkpoint.exists():
            saved = read_json(checkpoint)
            if (saved['input_identity'] != identity or saved.get('selector_policy') != 'single_full_window'
                    or saved['checkpoint_hash'] != content_hash({k: v for k, v in saved.items() if k != 'checkpoint_hash'})):
                raise ValueError('resumed full-window slate changed')
            self._reserve_checkpoint(saved, supply, slate)
            self._write_selection_views(saved, directory)
            return saved
        frozen = self._selector_inputs(supply, slate, design)
        payload = frozen['payload']
        requested = frozen['selection_eligible']
        response, routine, deferred, dropped = None, [], [], []
        if requested:
            template, version, digest = load_prompt(Path(self.config['_repository_root']) / 'prompts/astra_full_window_selector.md')
            selector_id = f'window-editor-{slate:02d}'
            prompt = render_prompt(template, {'CANDIDATES_WITH_REFERENCES': [payload['packages'][key] for key in payload['proposal_ids']],
                'ACCEPTED_BANK_DIGEST': payload['accepted_bank_digest'], 'IN_FLIGHT_RESERVATIONS': payload['in_flight_reservations'],
                'REJECTED_MEMORY_DIGEST': payload['rejected_memory_digest'], 'MAX_SELECTED': 16, 'SELECTOR_ID': selector_id})
            def validate(value):
                parse_window(value)
                if (value['selector_id'] != selector_id or
                        {r['candidate_id'] for r in value['candidate_assessments']} != set(payload['proposal_ids'])):
                    raise ValueError('full-window selector changed its frozen membership or identity')
            response = self._invoke(purpose='full_window_selection', stage=WINDOW_STAGE, call_id=f'window-{slate:02d}',
                prompt=prompt, version=version, digest=digest, parent_ids=payload['proposal_ids'], validator=validate)
            routine, deferred, dropped = frontier(supply, response, slate, self.config['selection'])
        value = {'schema_version': 1, 'slate_index': slate, 'input_identity': identity,
            'selector_policy': 'single_full_window', 'selection_completed': requested,
            'terminal_status': 'insufficient_eligible_window' if not requested else 'selected_slate_frozen' if routine else 'no_development_selected',
            'planning_started': False, 'source_labels_hidden_from_selector': True,
            'empty_selection_reason': ('exact_conflicts_leave_fewer_than_required_eligible_candidates' if not requested else 'no_eligible_qualified_candidates') if not routine else None,
            'exact_conflicts': deepcopy(frozen['exact_conflicts']), 'whole_development_slate_reserved_before_planning': True,
            'full_selector_selected_count': len(response['selected']) if response else 0,
            'source_ceiling_drop_ids': sorted(dropped), 'development_count': len(routine), 'deferred_count': len(deferred),
            'deferred_status': DEFERRED, 'source_rows': routine, 'deferred_rows': deferred,
            'specialized_selected_counts': dict(Counter(r['private_source'] for r in routine if r['private_source'] in SPECIAL_SOURCES)),
            'frozen_window': {'proposal_ids': payload['proposal_ids'], 'shards': payload['shards'],
                'input_snapshot_path': str((directory / 'selector_inputs.blinded.json').resolve()),
                'input_snapshot_sha256': content_hash((directory / 'selector_inputs.blinded.json').read_bytes())},
            'shard_selectors': [], 'shard_status': 'not_requested', 'global_selector': None,
            'full_window_selector': response, 'full_window_status': 'complete' if requested else 'not_requested_ineligible_window'}
        self._reserve_checkpoint(value, supply, slate)
        value['checkpoint_hash'] = content_hash(value)
        legacy._immutable(checkpoint, value, 'full-window selected slate')
        self._write_selection_views(value, directory)
        return value
