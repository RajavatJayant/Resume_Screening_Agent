"""
ranker.py
=========
PURPOSE: Sort candidates by ATS score — highest first.

INTERVIEW EXPLANATION:
"After scoring all candidates, I sort them in descending order
so the best match appears at rank #1. Python's sorted() with
reverse=True handles this in one line."
"""


def rank_candidates(scores: dict) -> list:
    """
    Sort candidates from best to worst ATS score.

    Args:
        scores: {
            "John Doe": {"score": 78, "breakdown": {...}, "resume_text": "..."},
            "Jane Doe": {"score": 55, ...},
        }

    Returns:
        Sorted list of (name, data) tuples — highest score first.
    """
    ranked = sorted(
        scores.items(),
        key=lambda item: item[1]["score"],
        reverse=True   # Descending: 90 → 78 → 55
    )
    return ranked
