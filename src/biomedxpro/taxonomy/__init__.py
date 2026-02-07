"""
Taxonomic Evolutionary Solver module.

This module implements the hierarchical decision tree architecture for BiomedXPro,
transforming flat multiclass classification into a series of focused binary decisions.
"""

from biomedxpro.taxonomy.artifact_store import JSONArtifactStore
from biomedxpro.taxonomy.dataset_slicer import DatasetSlicer
from biomedxpro.taxonomy.solver import TaxonomicSolver

__all__ = [
    "JSONArtifactStore",
    "DatasetSlicer",
    "TaxonomicSolver",
]
