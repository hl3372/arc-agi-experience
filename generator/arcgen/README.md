# arcgen

| Path | What it is |
|---|---|
| `paths.py` | Where the code is and where data goes (workspace, bank, cache). |
| `batch/` | The batch pipeline and its command line (see `batch/README.md`). |
| `lib/` | The shared library: Codex transport and call ledgers, contracts and schemas, the candidate DAG, evidence gates, palette, validation, bank registry and dossier tools. |
| `astra_*`, `sol_routine_*` | Earlier pipeline layers the batch pipeline builds on, named by the date they were introduced: contract bindings and repairs (`astra_usage_diversity_20260913`), plan and evidence structure checks (`astra_evidence_runtime_revision_20260911`), selection and admission (`astra_targeted_followup_20260911`, `astra_repetition_quality_20260912`), the native runtime and call wrapper (`astra_broad_arm_program_20260908`), the technical re-check (`sol_routine_astra_escalation_20260914`). |
| `prompts/`, `config/` | Base prompt templates and the base factory configuration the layers build on. |
