"""
Tau-Helper: Warrior Tau-Bench Helper Library

A comprehensive toolkit for creating, validating, and managing tasks, tools, and SOPs
across all Tau-Bench domains.

Main features:
- Instruction evaluation (user-facing vs procedural)
- Task generation and validation
- Tool verification
- SOP compliance checking
"""

__version__ = "0.1.0"
__author__ = "Warrior Tau-Bench Team"

from .llm import get_llm_client
from .evaluator import InstructionEvaluator

__all__ = ["get_llm_client", "InstructionEvaluator"]

