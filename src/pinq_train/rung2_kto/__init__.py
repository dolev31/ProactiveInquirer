"""Rung 2, RC4 -- KTO over the UNPAIRED signals the DPO rung's pairs file discards.

A preference pair is two rows to KTO, not one: the chosen side is a desirable (state, action)
and the rejected side an undesirable one, and the pairing itself is thrown away. That buys the
28,613 "an ASK here was wrong, at a state gold says was finished" signals the headline can only
express as a contrast against one 18-byte constant -- and it pays for them with the `V(s_t)`
variance the same-state contrast cancels exactly. Which side of that trade wins is what the arm
measures. See `convert.py` for the direction rule and `train.py` for the reference policy.

`pi train kto` is a separate task; nothing here registers a CLI.
"""

from .convert import (
    ON_CONFLICT,
    PAIR_CHOSEN,
    PAIR_REJECTED,
    PROVENANCE_FIELDS,
    SFT_ROW,
    SOURCE_KINDS,
    SYNTH_ASK_CHOSEN,
    SYNTH_ASK_REJECTED,
    SYNTH_STOP_CHOSEN,
    SYNTH_STOP_REJECTED,
    ConflictingLabels,
    NonCanonicalStop,
    UnparseableAction,
    balanced_weights,
    build_kto_rows,
    state_sha,
)
from .train import (
    KTO_LOSS_TYPES,
    MASS_RATIO_TOL,
    ImbalancedMasses,
    KTOConfig,
    NoKTORows,
    TooFewUndesirable,
    kto_argument_kwargs,
    mass_report,
    preflight,
    rows_for,
    rows_report,
    train,
)

__all__ = [
    "KTO_LOSS_TYPES",
    "MASS_RATIO_TOL",
    "ON_CONFLICT",
    "PAIR_CHOSEN",
    "PAIR_REJECTED",
    "PROVENANCE_FIELDS",
    "SFT_ROW",
    "SOURCE_KINDS",
    "SYNTH_ASK_CHOSEN",
    "SYNTH_ASK_REJECTED",
    "SYNTH_STOP_CHOSEN",
    "SYNTH_STOP_REJECTED",
    "ConflictingLabels",
    "ImbalancedMasses",
    "KTOConfig",
    "NoKTORows",
    "NonCanonicalStop",
    "TooFewUndesirable",
    "UnparseableAction",
    "balanced_weights",
    "build_kto_rows",
    "kto_argument_kwargs",
    "mass_report",
    "preflight",
    "rows_for",
    "rows_report",
    "state_sha",
    "train",
]
