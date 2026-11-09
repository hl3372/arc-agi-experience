# arcgen/batch: the batch pipeline

The user commands are in `cli.py` (`bin/arcgen`); the repository README describes them. This file maps the modules.

| Stage | Modules |
|---|---|
| Configuration | `config.py` (profile schema, validation, the single account, worker split), `routing.py` (model and effort per call), `slots.py` (property slots), `rungs.py`, `lines.py`, `structures.py` (optional size rungs and property lines) |
| Design | `allocation.py` (windows, calls, modifier assignment, budget ceilings), `allocations/make_allocation.py` (allocation table from a compact spec) |
| Preparation | `next_batch.py` (prepare, qualify, ready, authorize, run), `bank_inputs.py` (bank snapshot and reference corpus), `retrieval.py` (embeddings and nearest-bank search), `qualification.py`, `preflight.py` (account activity, route probes), `quota.py` (usage readings) |
| Ideas and selection | `idea_v1.py` (sketch, explore and commit turns with bank neighbours), `supply.py` (idea requests, selector inputs and request), `waves.py` (in-batch references between reference waves), `recent.py` (selector memory), `saved_queue.py`, `saved_ideas.py` (carried-over windows) |
| Scheduling | `coordinator.py`, `scheduler.py`, `budget.py`, `pipeline.py` |
| Development | `admission.py` (routes and per-idea draws), `contracts.py` (prompt sets and bindings), `runtime.py`, `downstream.py`, `gate.py` (evidence gate, blind, adjudication), `repair.py`, `repair_v1.py`, `review_budget.py`, `pairing.py` |
| Reporting and bank | `report.py`, `batch_report.py`, `properties.py`, `gallery.py`, `accounting.py`, `commit.py`, `commit_bank.py` |

Prompt texts of the reference design are in `prompts/v1/`; `prompts/` and `cards/` hold the base templates the planner,
realizer and checkers compose with them.
