import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from biomedxpro.core.domain import Population
from biomedxpro.core.interfaces import IHistoryRecorder


class HistoryRecorder(IHistoryRecorder):
    """
    The Scribe.
    Responsible for recording the entire evolutionary history to an append-only ledger.
    """

    def __init__(self, experiment_name: str, log_dir: str = "logs") -> None:
        """
        Initializes the history file.
        Filename format: {experiment_name}_{timestamp}_history.jsonl
        """
        # We generate the timestamp here to match the run's start time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}_history.jsonl"

        self.file_path = Path(log_dir) / filename
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure file exists
        self.file_path.touch()

    def record_generation(self, islands: Sequence[Population]) -> None:
        """
        Dumps the state of every individual in every island.
        """
        records = []
        for island in islands:
            for ind in island.individuals:
                # Base structure matches the Individual object
                record = {
                    "generation": island.generation,
                    "concept": island.concept,
                    "id": str(ind.id),
                    "parents": [str(p) for p in ind.parents],
                    # The Genotype (Input)
                    "negative_prompt": ind.genotype.negative_prompt,
                    "positive_prompt": ind.genotype.positive_prompt,
                    # The Results (Output) - Dumping the full dictionary
                    "metrics": ind.metrics if ind.metrics else None,
                    "metadata": ind.metadata,
                    "generation_born": ind.generation_born,
                }

                # Serialize to JSON Line
                records.append(json.dumps(record))

        # Batch write for performance
        if records:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write("\n".join(records) + "\n")
