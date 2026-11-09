"""Usage readings: a transient failure of the read is retried; only repeated failure pauses the account."""
import pytest
from arcgen.batch import quota as q

READING = {'account': 'main', 'weekly_remaining_percent': 51, 'weekly_used_percent': 49, 'weekly_resets_at': 0}


def observer(failures):
    calls = []
    def read(account, executable):
        calls.append(account)
        if len(calls) <= failures:
            raise RuntimeError('native quota read timed out')
        return READING
    return read, calls


def test_transient_read_failure_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(q.time, 'sleep', lambda s: None)
    read, calls = observer(2)
    assert q.Quota(tmp_path, 'codex', observer=read, account='main').start() == READING
    assert len(calls) == 3 and not (tmp_path / 'PAUSED_USAGE_main.json').exists()


def test_repeated_read_failure_pauses(tmp_path, monkeypatch):
    monkeypatch.setattr(q.time, 'sleep', lambda s: None)
    read, calls = observer(3)
    with pytest.raises(q.UsagePaused):
        q.Quota(tmp_path, 'codex', observer=read, account='main').observe(force=True)
    assert len(calls) == 3 and 'timed out' in (tmp_path / 'PAUSED_USAGE_main.json').read_text()


def test_failed_reading_keeps_a_recent_good_one(tmp_path, monkeypatch):
    monkeypatch.setattr(q.time, 'sleep', lambda s: None)
    calls = []
    def read(account, executable):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError('native quota read failed')
        return READING
    quota = q.Quota(tmp_path, 'codex', observer=read, account='main')
    assert quota.observe(force=True) == READING                       # good reading
    assert quota.observe(force=True) == READING                       # three failures, kept the recent good reading
    assert not (tmp_path / 'PAUSED_USAGE_main.json').exists()
    quota.checked -= 3600                                             # the good reading is now too old
    with pytest.raises(q.UsagePaused):
        quota.observe(force=True)


def test_with_usage_credits_an_exhausted_weekly_limit_does_not_pause(tmp_path, monkeypatch):
    monkeypatch.setattr(q, 'CREDITS', True)
    spent = {**READING, 'weekly_remaining_percent': 0, 'weekly_used_percent': 100}
    quota = q.Quota(tmp_path, 'codex', observer=lambda a, e: spent, account='main')
    assert quota.start() == spent
    quota.guard('planning', 'c1')  # no pause on the reading
    assert not (tmp_path / 'PAUSED_USAGE_main.json').exists()
    with pytest.raises(q.UsagePaused):  # a real refusal from the provider still pauses
        quota.failed('planning', 'c1', RuntimeError('You have hit your usage limit'))


def test_without_credits_an_exhausted_weekly_limit_pauses(tmp_path, monkeypatch):
    monkeypatch.setattr(q, 'CREDITS', False)
    spent = {**READING, 'weekly_remaining_percent': 0, 'weekly_used_percent': 100}
    with pytest.raises(q.UsagePaused):
        q.Quota(tmp_path, 'codex', observer=lambda a, e: spent, account='main').start()
