"""Separate research review panels are disabled by design in this pipeline.

The scheduler keeps its research slots (always empty); these entry points refuse any research job so a design or
resume can never schedule one.
"""
from .common import IntegrityError


def _disabled(*args, **kwargs):
    raise IntegrityError('separate research is disabled by design')


class CanonicalPanels:
    def __init__(self, *args, **kwargs):
        _disabled()


def frozen_anchors(root):
    _disabled()


def run_job(*args, **kwargs):
    _disabled()


def validate_jobs(jobs, windows, pools, phase, *, anchors=None):
    if jobs:
        _disabled()
    return []
