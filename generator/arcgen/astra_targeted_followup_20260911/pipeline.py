"""Dispatch control shared by the pipeline layers."""


class DispatchPaused(BaseException):
    """Control interruption, not a failed task or a fatal native DAG defect.

    Like a process interruption, leave the active stage to native reconciliation.
    Never rewrite a fatal-stage journal or reset a consumed attempt on resume.
    """
