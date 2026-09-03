"""Collectors and the pretenuring policies that distinguish them."""

from .base import Collector, COPY_COST, SWEEP_COST
from .marksweep import MarkSweep
from .semispace import SemiSpace
from .generational import Generational
from .predictors import (
    Predictor, NeverPretenure, AlwaysPretenure, PerfectOracle, SiteTable,
    LearnedPredictor,
)

__all__ = [
    "Collector", "COPY_COST", "SWEEP_COST",
    "MarkSweep", "SemiSpace", "Generational",
    "Predictor", "NeverPretenure", "AlwaysPretenure", "PerfectOracle",
    "SiteTable", "LearnedPredictor",
]
