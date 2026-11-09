"""ARC-style bundles for packing_v2 (DESIGN.md sections 8 and 13.10).

* ``grids``       grid analysis shared by the rules (background, objects, caches)
* ``transforms``  the grid -> grid rule library (71 families; 51 generated, 20 rivals only; general wrap shifts in ``rivals``) and ``Rule``
* ``scalars``     grid -> number / colour / size rules and their certificate
* ``rivals``      the elimination set and ``certify`` / ``unique_test_output``
* ``gen``         rule proposals, borrowed pictures and drawn scenes, sibling readings
* ``aux_rules``   the aux tasks themselves as rules (stated and hidden modes)
* ``questions``   the questions an ARC part asks
* ``bundle``      ``arc_part(rng, mode=None) -> Part``
"""
