"""Supervised fine-tuning: data pipeline (teacher attempts -> rows -> cache -> tokens) and the FSDP2 full-FT trainer.

Modules are run as scripts, e.g. `python -m arcsolver.sft.build_cache ...` or
`torchrun --nproc_per_node 8 -m arcsolver.sft.train_fsdp ...`; see README.md in this directory.
"""
