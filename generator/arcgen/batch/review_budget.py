"""Native slate-budget interface backed by a shared durable conditional pool."""
class SharedSlateBudget:
    def __init__(self,runtime,slate,limit):
        self.runtime=runtime;self.slate=str(slate);self.limit=int(limit)
        self.sync()

    def sync(self):
        with self.runtime.lock:
            rows=self.runtime.budget.review_snapshot(self.runtime.window)
            self.runtime.state.setdefault('e2_escalation_budgets',{})[self.slate]={
                'limit':self.limit,'policy':'shared-conditional-packages-v1',
                'used':sum(r['charged']+r['remaining'] for r in rows),
                'reservations':{r['key'].split('/',1)[1]:r['requested'] for r in rows},
                'packages':rows}
            self.runtime.save()

    def reserve_once(self,key,count):
        result=self.runtime.budget.reserve_review(window=self.runtime.window,
            account=self.runtime.factory_config['run']['account'],phase=self.runtime.phase,key=key,count=count)
        self.sync();return result

    def release_task(self,task_id):
        self.runtime.budget.release_reviews(self.runtime.window,task_id);self.sync()

    @property
    def remaining(self):
        # Kept for native diagnostics; the durable admission check is authoritative.
        s=self.runtime.budget.summary();phase=self.runtime.budget.phase(self.runtime.phase)
        spent=sum(c['category']=='conditional_evidence_repair' for c in s['calls'])
        return max(0,phase['categories']['conditional_evidence_repair']-spent)
