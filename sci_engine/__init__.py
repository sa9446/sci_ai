"""Scientific AI Model Engine — Neuro-Symbolic Computation Framework.

Architecture:
  NeuralSymbolicRouter  →  InsulatedSandbox  →  ResultGroundingValidator
  (LLM structured parse)   (exec isolation)    (physics & dimensional check)
"""
from .agent import NeuralSymbolicRouter, ScientificAnalysis
from .executor import InsulatedSandbox, ExecutionResult
from .validator import ResultGroundingValidator, ValidationReport
from .config import settings

__version__ = "1.0.0"
__all__ = [
    "NeuralSymbolicRouter",
    "ScientificAnalysis",
    "InsulatedSandbox",
    "ExecutionResult",
    "ResultGroundingValidator",
    "ValidationReport",
    "settings",
]
