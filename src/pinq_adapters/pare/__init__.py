"""PARE adapter (deepakn97/pare) — Grid E, the transfer experiment.

Importable with PARE absent: upstream is only ever imported inside function bodies,
guarded by `available()`.
"""

from ._probe import APPS, N_SCENARIOS, SPLIT, app_of, available, load_scenario_ids
from .actuator import PareActuator
from .suite import (
    ARM_BASELINE,
    ARM_INQUIRER_PROMPTED,
    ARM_VERBOSITY,
    CORPUS_ID,
    GRID_E_ARMS,
    AppStateRetriever,
    PareSuite,
    PareSuiteUnavailable,
)

__all__ = [
    "APPS",
    "ARM_BASELINE",
    "ARM_INQUIRER_PROMPTED",
    "ARM_VERBOSITY",
    "CORPUS_ID",
    "GRID_E_ARMS",
    "N_SCENARIOS",
    "SPLIT",
    "AppStateRetriever",
    "PareActuator",
    "PareSuite",
    "PareSuiteUnavailable",
    "app_of",
    "available",
    "load_scenario_ids",
]
