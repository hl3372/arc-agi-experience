"""Auto-load second-batch core generator modules.

Each batch module in this package registers task generators via registry.register.
The package importer keeps the central registry stable while allowing batch
implementations to live in separate files.
"""
from __future__ import annotations

import importlib
import pkgutil


for module_info in pkgutil.iter_modules(__path__):
    if module_info.name.startswith("_") or module_info.name == "utils":
        continue
    importlib.import_module(f"{__name__}.{module_info.name}")
