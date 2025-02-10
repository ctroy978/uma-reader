# progression.py
from enum import Enum
from typing import Optional


class QuestionCategory(str, Enum):
    LITERAL_BASIC = "literal_basic"
    LITERAL_DETAILED = "literal_detailed"
    INFERENTIAL_SIMPLE = "inferential_simple"
    INFERENTIAL_COMPLEX = "inferential_complex"


# Define progression path
PROGRESSION_ORDER = [
    QuestionCategory.LITERAL_BASIC,
    QuestionCategory.LITERAL_DETAILED,
    QuestionCategory.INFERENTIAL_SIMPLE,
    QuestionCategory.INFERENTIAL_COMPLEX,
]


def get_next_category(current_category: str) -> Optional[str]:
    """Get the next category in the progression path"""
    try:
        current_idx = PROGRESSION_ORDER.index(QuestionCategory(current_category))
        if current_idx < len(PROGRESSION_ORDER) - 1:
            return PROGRESSION_ORDER[current_idx + 1].value
        return current_category  # Stay at highest if already at top
    except (ValueError, IndexError):
        return QuestionCategory.LITERAL_BASIC.value


def get_previous_category(current_category: str) -> Optional[str]:
    """Get the previous category in the progression path"""
    try:
        current_idx = PROGRESSION_ORDER.index(QuestionCategory(current_category))
        if current_idx > 0:
            return PROGRESSION_ORDER[current_idx - 1].value
        return current_category  # Stay at lowest if already at bottom
    except (ValueError, IndexError):
        return QuestionCategory.LITERAL_BASIC.value


def update_category_on_result(assessment, is_correct: bool) -> None:
    """Update category based on answer correctness"""
    if is_correct:
        next_cat = get_next_category(assessment.current_category)
        assessment.current_category = next_cat
    else:
        prev_cat = get_previous_category(assessment.current_category)
        assessment.current_category = prev_cat
