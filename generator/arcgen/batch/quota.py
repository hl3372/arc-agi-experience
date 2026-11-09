"""Read-only usage observations of the Codex account and resumable, bounded usage pauses.

A reading starts a short-lived `codex app-server --stdio` with the account's CODEX_HOME and asks for its rate limits
(`account/rateLimits/read`); it makes no model call and reads no credential file.
"""
from pathlib import Path
import json, os, selectors, subprocess, threading, time
from .common import ACCOUNT, HOMES, IntegrityError, atomic, frozen, read
from .config import OPERATIONS
CREDITS = OPERATIONS.get('usage_credits', False)  # past the weekly limit the account runs on credits; only a provider refusal pauses


def account_limits(account, executable, timeout=35):
    account_env=dict(os.environ);account_env['CODEX_HOME']=str(HOMES[account])  # the configured account's home
    process=subprocess.Popen([executable,'app-server','--stdio'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=account_env)
    def send(value):process.stdin.write((json.dumps(value)+'\n').encode());process.stdin.flush()
    selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
    send({'method':'initialize','id':0,'params':{'clientInfo':{'name':'arcgen_quota_observer','version':'1.0.0'}}})
    end=time.monotonic()+timeout;buf=b''
    try:
        while time.monotonic()<end:
            for key,_ in selector.select(min(1,max(0,end-time.monotonic()))):
                data=os.read(key.fileobj.fileno(),65536)
                if not data:raise IntegrityError('quota observer ended before returning limits')
                buf+=data
                while b'\n' in buf:
                    line,buf=buf.split(b'\n',1)
                    try:r=json.loads(line)
                    except ValueError:continue
                    if r.get('id')==0:
                        if 'error' in r:raise IntegrityError('quota initialization failed')
                        send({'method':'initialized'});send({'method':'account/rateLimits/read','id':1})
                    elif r.get('id')==1:
                        if 'error' in r:raise IntegrityError('native quota read failed')
                        result=r['result'];buckets=result.get('rateLimitsByLimitId') or {'codex':result['rateLimits']}
                        value=buckets.get('codex')
                        if not value:raise IntegrityError('shared Codex quota bucket unavailable')
                        windows=[value.get(k) for k in ('primary','secondary') if value.get(k)]
                        weekly=next((w for w in windows if w.get('windowDurationMins')==10080),None)
                        if weekly is None:raise IntegrityError('weekly Codex quota not reported')
                        return {'account':account,'observed_at_epoch':time.time(),'limit_id':'codex',
                            'windows':windows,'weekly_used_percent':weekly['usedPercent'],
                            'weekly_remaining_percent':100-weekly['usedPercent'],'weekly_resets_at':weekly['resetsAt']}
        raise IntegrityError('native quota read timed out')
    finally:
        selector.close();process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:process.kill();process.wait()




class UsagePaused(BaseException):
    """Control flow, not a terminal semantic/transport failure to be swallowed."""


class Quota:
    def __init__(self, root, executable, observer=account_limits, mock=False, account=ACCOUNT):
        if account not in HOMES: raise IntegrityError("unknown quota account")
        self.account=account
        self.root=Path(root);self.executable=executable;self.observer=observer;self.mock=mock
        self.lock=threading.RLock();self.latest=None;self.checked=0

    def observe(self, force=False):
        with self.lock:
            if self.mock: return {'account':self.account,'weekly_remaining_percent':100,'weekly_used_percent':0,'weekly_resets_at':0}
            if not force and self.latest is not None and time.monotonic()-self.checked < 45: return self.latest
            for attempt in range(3):  # the read starts a short-lived Codex app-server; transient failures are retried
                try:
                    value=self.observer(self.account,self.executable); break
                except Exception as exc:
                    if attempt==2:
                        # A failed reading is not a limit: keep using a good reading under 30 minutes old that shows
                        # real headroom (an actual limit is still caught from the calls' own errors).
                        if self.latest is not None and time.monotonic()-self.checked < 1800 and (CREDITS or self.latest['weekly_remaining_percent'] > 5):
                            frozen(self.root/'quota'/self.account/f'{time.time_ns()}-read-failed.json',
                                   {'error_type':type(exc).__name__,'error':str(exc)[:300],'kept_reading':self.latest})
                            return self.latest
                        self.pause('quota_observation_unavailable',error_type=type(exc).__name__,error=str(exc)[:300])
                    time.sleep(5*(attempt+1))
            self.latest=value;self.checked=time.monotonic()
            frozen(self.root/'quota'/self.account/f'{time.time_ns()}.json',value)
            atomic(self.root/'quota'/self.account/'latest.json',value)
            return value

    def pause(self, reason, **details):
        with self.lock:
            atomic(self.root/('PAUSED_USAGE_'+self.account+'.json'),{'reason':reason,'at_epoch':time.time(),
                'last_observation':self.latest,'account':self.account,**details})
        raise UsagePaused(reason)

    def start(self):
        value=self.observe(force=True);path=self.root/('PAUSED_USAGE_'+self.account+'.json')
        if value['weekly_remaining_percent'] <= 0 and not CREDITS: self.pause(self.account+'_usage_exhausted')
        if path.exists():
            previous=read(path);old=previous.get('last_observation') or {}
            reset=(value['weekly_remaining_percent']>old.get('weekly_remaining_percent',-1)
                or abs(value.get('weekly_resets_at',0)-old.get('weekly_resets_at',0))>60
                or previous['reason']=='quota_observation_unavailable'
                or CREDITS)  # with credits an explicit resume clears a usage pause (the owner chose to continue)
            if not reset: raise UsagePaused('usage pause remains; reset has not been observed')
            frozen(self.root/'quota'/self.account/f'{time.time_ns()}-resume.json',{'previous_pause':previous,'new_observation':value,
                'quota_reset_observed':True,'budgets_refilled':False})
            path.unlink()
        return value

    def guard(self, stage, call_id):
        if (self.root/('PAUSED_USAGE_'+self.account+'.json')).exists(): raise UsagePaused(self.account+' is paused')
        value=self.observe()
        if value['weekly_remaining_percent'] <= 0 and not CREDITS: self.pause(self.account+'_usage_exhausted')

    def failed(self, stage, call_id, exc):
        message=str(exc).lower()
        if any(s in message for s in ('usage limit','usage_limit','usage_limit_reached','rate limit reached','rate_limit_exceeded',
                                      'weekly limit','insufficient_quota','you have hit your')):
            self.pause('provider_usage_limit',stage=stage,call_id=call_id)
