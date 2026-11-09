"""The assembler: render any list of producer Parts into one diverse (prompt, target) row.

    from pv2.assemble import assemble_row, AssembleConfig
    row = assemble_row(np.random.default_rng(seed), parts, config=AssembleConfig())
    row.prompt, row.target, row.meta

Modules: ``refs`` (reference schemes and the resolver), ``layouts`` (placements, delimiters, scoped
sentences), ``styles`` (answer styles: instruction / render / parse, the style x kind matrix),
``row`` (assemble_row).  ``pv2.validate.validate_row`` checks every row (V1-V5).
"""
from pv2.assemble.row import AssembleConfig, AssemblyError, Row, assemble_row, check_parts  # noqa: F401
