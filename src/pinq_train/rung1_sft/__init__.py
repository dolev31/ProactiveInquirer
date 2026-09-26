"""Rung 1 -- rejection-sampling SFT. LoRA, and a loss masked to the Inquirer's own tokens.

The mask is the rung: an SFT example here is a multi-thousand-token state followed by a short
JSON action, and an unmasked loss would spend better than 95% of its gradient teaching the
model to reproduce retrieved documents. See `mask.py`.
"""

from .collapse import DISTINCT3_FLOOR, DiversityReport, ModeCollapse, assert_diverse, distinct_n
from .curriculum import CURRICULA, SHUFFLED, assign_strata, order_examples
from .mask import (
    IGNORE_INDEX,
    MARGIN_SIGMAS,
    ChatTokenizer,
    MaskedExample,
    OverLength,
    TemplateNotPrefix,
    Tokenizer,
    build_chat_masked_example,
    build_masked_example,
    margin_threshold,
    pack,
    truncate_left,
)
from .train import (
    BuiltExamples,
    HeldOutDataset,
    LoraSpec,
    NotValidatedOnHardware,
    SFTConfig,
    accum_scale,
    assert_train_split,
    build_examples,
    dataset_report,
    estimate_gpu_hours,
    preflight,
    read_jsonl,
    train,
    weighted_loss,
)

__all__ = [
    "CURRICULA",
    "DISTINCT3_FLOOR",
    "IGNORE_INDEX",
    "MARGIN_SIGMAS",
    "SHUFFLED",
    "BuiltExamples",
    "ChatTokenizer",
    "DiversityReport",
    "HeldOutDataset",
    "LoraSpec",
    "MaskedExample",
    "ModeCollapse",
    "NotValidatedOnHardware",
    "OverLength",
    "SFTConfig",
    "TemplateNotPrefix",
    "Tokenizer",
    "accum_scale",
    "assert_diverse",
    "assert_train_split",
    "assign_strata",
    "build_chat_masked_example",
    "build_examples",
    "build_masked_example",
    "dataset_report",
    "distinct_n",
    "estimate_gpu_hours",
    "margin_threshold",
    "order_examples",
    "pack",
    "preflight",
    "read_jsonl",
    "train",
    "truncate_left",
    "weighted_loss",
]
