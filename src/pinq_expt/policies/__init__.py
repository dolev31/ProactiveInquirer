"""The Inquirer policies under test. One class per hypothesis, no registry.

Every class here satisfies `pinq.protocols.Inquirer` structurally: `policy_id`, `reset(view,
seed)`, `act(s)`. `act` takes State and nothing else, in all nine of them.
"""

from pinq_expt.policies.ancestors import IRCoTInquirer, SelfAskInquirer, SelfInquireInquirer
from pinq_expt.policies.base import MALFORMED, LLMPolicy
from pinq_expt.policies.controls import (
    ChecklistInquirer,
    GoldEvidenceInquirer,
    OracleVreqInquirer,
    ParallelReplayInquirer,
    RandomQInquirer,
    ScriptedInquirer,
    recorded_questions,
)
from pinq_expt.policies.prompted import (
    CapAwarePromptedInquirer,
    Depth1Inquirer,
    NoEvidenceInquirer,
    PromptedInquirer,
)
from pinq_expt.policies.structured_baselines import Par2RagInquirer

__all__ = [
    "MALFORMED",
    "ChecklistInquirer",
    "Depth1Inquirer",
    "GoldEvidenceInquirer",
    "IRCoTInquirer",
    "LLMPolicy",
    "NoEvidenceInquirer",
    "OracleVreqInquirer",
    "ParallelReplayInquirer",
    "CapAwarePromptedInquirer",
    "Par2RagInquirer",
    "PromptedInquirer",
    "RandomQInquirer",
    "ScriptedInquirer",
    "SelfAskInquirer",
    "SelfInquireInquirer",
    "recorded_questions",
]
