"""Two explicit factory recipes. Absence retains the original resolved config.

Review policy selects required evidence; it never authorizes publication.
"""
from copy import deepcopy
from .util import content_hash

REDUCED_PROFILE = "astra_reduced_v1"
AUTHORING_CONTRACT_PROFILE = "astra_authoring_contract_r2"
REDUCED_REVIEW = {
    "id": "astra_one_actual_board_v1", "version": 1,
    "board_indices": [1], "stage": "v4_editorial_board",
    "prompt_name": "editorial_board", "model": "gpt-6-astra",
    "reasoning_effort": "high", "cohort": "finite_completed",
    "group_size": 6, "maximum_parallel_groups": 4,
}
REDUCED_POLICY = {
    "profile": REDUCED_PROFILE, "version": 1,
    "selection": "single_full_window", "review": REDUCED_REVIEW,
    "mechanism_audits": "not_requested", "audit_feature_prevalence": None,
}
REDUCED_EFFORTS = {
    "idea_conditioned": "high", "idea_unconditioned": "xhigh",
    "idea_logical_leap": "xhigh", "idea_episode_joint": "xhigh",
    "shard_selection": "high", "global_selection": "high",
    "full_window_selection": "high", "planning": "xhigh", "realization": "high",
    "technical_verification": "high", "primary_blind": "xhigh",
    "conditional_blind": "xhigh", "ambiguity_adjudication": "high",
    "targeted_repair": "xhigh", "editorial_board": "high",
    "logical_leap_audit": "high", "structural_transfer_audit": "high",
    "mechanism_audit": "high", "diversity_audit": "high", "transport_probe": "high",
}


def pipeline_policy(config):
    name = config.get("pipeline_profile")
    if name is None:
        if "_pipeline_policy" in config:
            raise ValueError("pipeline policy requires an explicit profile")
        return None
    if name != REDUCED_PROFILE or content_hash(config.get("_pipeline_policy")) != content_hash(REDUCED_POLICY):
        raise ValueError("unknown or mismatched factory pipeline policy")
    return deepcopy(REDUCED_POLICY)


def review_policy(config):
    policy = pipeline_policy(config)
    return policy["review"] if policy else None


def validate_review_policy(policy):
    if policy is not None and content_hash(policy) != content_hash(REDUCED_REVIEW):
        raise ValueError("unknown or mismatched required-board policy")
    return deepcopy(policy)
