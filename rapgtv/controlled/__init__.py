"""Synthetic controlled-data generator used by the paper experiments."""

from .generator import ControlledCondition, ControlledRealization, generate_controlled_realization

__all__ = ["ControlledCondition", "ControlledRealization", "generate_controlled_realization"]
