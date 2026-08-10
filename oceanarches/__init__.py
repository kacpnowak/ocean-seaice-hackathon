"""OceanArches -- AI ocean and sea-ice modelling on GLORYS, built on top of geoarches.

Start here:  docs/00_start_here.md

Deliberately imports nothing.  ``import oceanarches`` costs about 1 ms; pulling
torch in here would make it 1.3 s for every script that only wants
``oceanarches.paths``.  The checkpoint allowlist that used to live here now sits
in :mod:`oceanarches.lightning_modules`, which every path that opens a
checkpoint goes through anyway and which imports torch regardless.
"""

__version__ = "0.1.0"
