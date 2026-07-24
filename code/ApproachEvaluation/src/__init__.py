# This is necessary to make the directory a package and to expose the main classes/functions
from .evaluator import (
    SummarizationEvaluator,
    ComparisonEvaluator,
    EvaluationMetrics
)

__all__ = [
    'SummarizationEvaluator',
    'ComparisonEvaluator',
    'EvaluationMetrics'
]

__version__ = '1.0.0'
