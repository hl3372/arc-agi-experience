"""The logical-call wrapper every pipeline call goes through: a terminal failure is recorded once, never retried here."""
from __future__ import annotations
from arcgen.lib.model import ModelInvocationError
from arcgen.lib.util import content_hash
from .common import IntegrityError, read, frozen


def invoke(runtime,kwargs):
    """A terminal logical failure is data, never a request for another retry."""
    p=runtime.run_dir/'opportunities'/kwargs['stage']/(kwargs['call_id']+'.json')
    identity=content_hash({k:v for k,v in kwargs.items() if k!='response_validator'})
    if p.exists():
        saved=read(p)
        if saved['request_hash']!=identity:raise IntegrityError('logical opportunity changed')
        if saved['value'] is None:return None
        # Re-enter the native successful cache path to verify every actual call
        # artifact, without another provider request.
    try:value=runtime.invoke_rendered(**kwargs).value
    except ModelInvocationError as exc:value=None
    frozen(p,{'request_hash':identity,'value':value,'value_hash':content_hash(value)})
    return value
