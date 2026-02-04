"""
Post-evolution analysis tools for BioMedXPro.

Load evolutionary history from JSONL logs and compute insights about:
- Fitness progression across generations
- Champion discovery and lineage tracking
- Convergence and diversity metrics
- Genetic operator effectiveness
"""

from biomedxpro.analysis.loader import EvolutionHistory, HistoryRecord, IslandSnapshot
from biomedxpro.analysis.metrics import (
    ChampionAnalyzer,
    ChampionInfo,
    ConvergenceDetector,
    ConvergenceInfo,
    FitnessProgressionAnalyzer,
    FitnessStats,
)
from biomedxpro.analysis.reports import AnalysisReport, generate_report
from biomedxpro.analysis.visualizations import (
    generate_all_plots,
    plot_champion_timeline,
    plot_convergence_heatmap,
    plot_fitness_curves,
    plot_improvement_rates,
)

__all__ = [
    # Loader
    "EvolutionHistory",
    "HistoryRecord",
    "IslandSnapshot",
    # Analyzers
    "FitnessProgressionAnalyzer",
    "ChampionAnalyzer",
    "ConvergenceDetector",
    # Data classes
    "FitnessStats",
    "ChampionInfo",
    "ConvergenceInfo",
    # Reports
    "AnalysisReport",
    "generate_report",
    # Visualizations
    "plot_fitness_curves",
    "plot_champion_timeline",
    "plot_convergence_heatmap",
    "plot_improvement_rates",
    "generate_all_plots",
]
