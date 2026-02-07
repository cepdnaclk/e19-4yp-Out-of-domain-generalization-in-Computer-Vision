import json
import shutil
import sys
from pathlib import Path

import torch
from loguru import logger

from biomedxpro.core.domain import (
    CreationOperation,
    EncodedDataset,
    EvolutionParams,
    Individual,
    PromptGenotype,
    TaskDefinition,
)
from biomedxpro.core.interfaces import ILLMClient
from biomedxpro.taxonomy.artifact_store import (
    JSONArtifactStore,
)  # You need to ensure this exists or use Memory
from biomedxpro.taxonomy.builder import LLMTaxonomyBuilder
from biomedxpro.taxonomy.solver import TaxonomicSolver

# --- MOCKS ---


class MockLLM(ILLMClient):
    """Returns a fixed Kidney Tree JSON."""

    def generate(self, prompt: str) -> str:
        # A valid asymmetric tree: Normal vs (Cyst vs Tumor)
        tree = {
            "node_id": "root",
            "group_name": "Global Analysis",
            "left_classes": ["Normal"],
            "right_classes": ["Cyst", "Tumor"],
            "children": {
                "left": None,  # Leaf (Normal is terminal)
                "right": {
                    "node_id": "pathology",
                    "group_name": "Lesion Type",
                    "left_classes": ["Cyst"],
                    "right_classes": ["Tumor"],
                },
            },
        }
        return json.dumps(tree)


class MockOrchestrator:
    """Simulates the evolutionary engine finding a good prompt."""

    def __init__(self, dataset, task) -> None:
        self.task = task
        self.dataset = dataset

    def run(self):
        logger.info(f"    [MockEngine] Evolving for task: {self.task.class_names}")
        logger.info(f"    [MockEngine] Dataset size: {self.dataset.num_samples}")

        # Return a fake "Winner"
        winner = Individual(
            id="winner_1",
            genotype=PromptGenotype(
                prompts=("Baseline", f"Find features of {self.task.class_names[1]}")
            ),
            generation_born=5,
            operation=CreationOperation.CROSSOVER,
            concept="Mock Concept",
            metadata={"score": 0.95},
        )

        # CRITICAL: Update metrics so the individual is considered evaluated
        winner.update_metrics(
            {
                "soft_f1_macro": 0.95,
                "accuracy": 0.93,
                "f1_macro": 0.94,
                "auc": 0.96,
                "f1_weighted": 0.94,
                "inverted_bce": 0.92,
            }
        )

        return [winner]


# --- MAIN EXECUTION ---


def main() -> None:
    # 1. Setup Workspace
    work_dir = Path("simulation_output")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

    # 2. Create Synthetic Data (100 samples, 4 classes)
    logger.info(">>> 1. Generating Synthetic Data...")
    features = torch.randn(100, 512)
    labels = torch.randint(0, 3, (100,))  # 0=Normal, 1=Cyst, 2=Tumor
    dataset = EncodedDataset(
        name="SynthKidney",
        features=features,
        labels=labels,
        class_names=["Normal", "Cyst", "Tumor"],
    )

    # 3. Architect the Tree
    logger.info(">>> 2. Building Taxonomy...")
    task = TaskDefinition("Kidney", "CT", dataset.class_names, "Rad", None)
    builder = LLMTaxonomyBuilder(MockLLM())
    root_node = builder.build_taxonomy(task)
    evolution_params = EvolutionParams(
        generations=10,
        target_metric="soft_f1_macro",
        island_capacity=100,
        initial_pop_size=10,
        num_parents=10,
        offspring_crossover=10,
        offspring_mutated=2,
    )
    # 4. Initialize Components
    store = JSONArtifactStore(base_path=str(work_dir / "artifacts"))

    # Factory that returns our Mock Engine
    def orchestrator_factory(ds, t) -> MockOrchestrator:
        return MockOrchestrator(ds, t)

    # 5. Run Solver
    logger.info(">>> 3. Running Taxonomic Solver...")
    solver = TaxonomicSolver(
        root_node=root_node,
        artifact_store=store,
        train_dataset=dataset,
        orchestrator_factory=orchestrator_factory,
        base_task_def=task,
        evolution_params=evolution_params,
    )
    solver.run()

    # 6. Verify Outputs
    logger.info("\n>>> 4. Verification <<<")

    # We expect artifacts for: "root" and "pathology"
    # "root" splits Normal vs (Cyst, Tumor)
    # "pathology" splits Cyst vs Tumor

    expected_nodes = ["root", "pathology"]
    for node_id in expected_nodes:
        path = work_dir / "artifacts" / node_id / "ensemble.json"
        if path.exists():
            logger.success(f"✅ Found artifact for {node_id}")
            with open(path) as f:
                data = json.load(f)
                logger.info(f"   - Metadata: {data['metadata']['label_mapping']}")
        else:
            logger.error(f"❌ Missing artifact for {node_id}")


if __name__ == "__main__":
    main()
