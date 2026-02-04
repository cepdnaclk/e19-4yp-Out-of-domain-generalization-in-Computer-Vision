"""
Post-evolution analysis tools for BioMedXPro.

Load evolutionary history from JSONL logs and compute insights about:
- Fitness progression across generations
- Champion discovery and lineage tracking
- Convergence and diversity metrics
- Genetic operator effectiveness
"""

from biomedxpro.analysis.loader import EvolutionHistory, HistoryRecord, IslandSnapshot

__all__ = [
    "EvolutionHistory",
    "HistoryRecord",
    "IslandSnapshot",
]
