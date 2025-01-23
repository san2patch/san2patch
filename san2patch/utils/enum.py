from enum import Enum, auto
from typing import Literal


class LanguageEnum(Enum):
    C = auto()


class ExperimentStepEnum(Enum):
    START = "start"
    GENPATCH = "genpatch"
    EVAL = "eval"
    PATCH = "patch"
    BUILD = "build"
    VULN_TEST = "vuln_test"
    FUNC_TEST = "func_test"
    END = "end"


class ExperimentResEnum(Enum):
    EXCEPTION_RAISED = "exception_raised"
    GENPATCH_FAILED = "genpatch_failed"
    PATCH_FAILED = "patch_failed"
    BUILD_FAILED = "build_failed"
    VULN_FAILED = "vuln_test_failed"
    FUNC_FAILED = "func_test_failed"
    SUCCESS = "success"


SELECT_METHODS = Literal["sample", "greedy"]
# "zero_zero": (ZERO, ZERO, ZERO),
# "zero_medium": (ZERO, MEDIUM, MEDIUM),
# "zero_high": (ZERO, HIGH, HIGH),
# "medium_zero": (MEDIUM, ZERO, ZERO),
# "medium_medium": (MEDIUM, MEDIUM, MEDIUM),
# "medium_high": (MEDIUM, HIGH, HIGH),
# "high_zero": (HIGH, ZERO, ZERO),
# "high_medium": (HIGH, MEDIUM, MEDIUM),
# "high_high": (HIGH, HIGH, HIGH),
TEMPERATURE_SETTING = Literal[
    "zero",
    "low",
    "medium",
    "high",
    "zero_zero",
    "zero_medium",
    "zero_high",
    "medium_zero",
    "medium_medium",
    "medium_high",
    "high_zero",
    "high_medium",
    "high_high",
]
CODE_CONTEXT_MODE = Literal["line", "ast"]


VERSION_LIST = Literal["tot", "zeroshot", "cot", "no_context", "only_context"]


MODEL_LIST = Literal[
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5",
    "claude-3.5-sonnet",
    "claude-3-haiku",
    "claude-3-opus",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]
