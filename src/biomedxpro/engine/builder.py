import sys

from loguru import logger

from biomedxpro.core.domain import DataSplit, EncodedDataset
from biomedxpro.engine.config import MasterConfig
from biomedxpro.engine.orchestrator import Orchestrator
from biomedxpro.impl.adapters import get_adapter
from biomedxpro.impl.data_loader import BiomedDataLoader
from biomedxpro.impl.evaluator import FitnessEvaluator
from biomedxpro.impl.llm_client import create_llm_client
from biomedxpro.impl.llm_operator import LLMOperator
from biomedxpro.impl.selection import RouletteWheelSelector
from biomedxpro.utils.history import HistoryRecorder
from biomedxpro.utils.token_logging import TokenUsageLogger

def load_datasets(
    config: MasterConfig,
) -> tuple[EncodedDataset, EncodedDataset, EncodedDataset]:
    """
    Factory method for Data Layer artifacts.
    Handles Adapter resolution, raw loading, and GPU encoding/caching.
    """
    logger.info(f"Initializing adapter: {config.dataset.adapter}")
    try:
        adapter = get_adapter(
            config.dataset.adapter,
            root=config.dataset.root,
            shots=config.dataset.shots,
            **config.dataset.adapter_params,
        )
    except Exception as e:
        logger.error(f"Failed to initialize adapter: {e}")
        sys.exit(1)

    loader = BiomedDataLoader(
        cache_dir=config.dataset.cache_dir,
        batch_size=config.execution.batch_size,
        device=config.execution.device,
        num_workers=config.execution.dataloader_cpu_workers,
    )

    logger.info("Encapsulating data into optimized EncodedDataset artifacts...")
    train_ds = loader.load_encoded_dataset(
        name=f"{config.dataset.name}_train",
        samples=adapter.load_samples(DataSplit.TRAIN),
        class_names=config.dataset.class_names,
    )

    val_ds = loader.load_encoded_dataset(
        name=f"{config.dataset.name}_val",
        samples=adapter.load_samples(DataSplit.VAL),
        class_names=config.dataset.class_names,
    )

    test_ds = loader.load_encoded_dataset(
        name=f"{config.dataset.name}_test",
        samples=adapter.load_samples(DataSplit.TEST),
        class_names=config.dataset.class_names,
    )

    return train_ds, val_ds, test_ds


def build_orchestrator(
    config: MasterConfig,
    train_ds: EncodedDataset,
    val_ds: EncodedDataset,
    recorder: HistoryRecorder,
    token_logger: TokenUsageLogger 
) -> Orchestrator:
    """
    Dependency Injection container for the Engine.
    """
    logger.info("Bootstrapping evolutionary components...")
    
    if token_logger is None:
        raise ValueError("TokenUsageLogger must be provided to build_orchestrator")


    llm_client = create_llm_client(config.llm, token_logger=token_logger)

    operator = LLMOperator(
        llm=llm_client, strategy=config.strategy, task_def=config.task
    )

    # We create the evaluator here and inject it into the Orchestrator
    evaluator = FitnessEvaluator(
        device=config.execution.device, batch_size=config.execution.batch_size
    )

    selector = RouletteWheelSelector()

    return Orchestrator(
        evaluator=evaluator,
        operator=operator,
        selector=selector,
        train_dataset=train_ds,
        val_dataset=val_ds,
        params=config.evolution,
        recorder=recorder,
        exec_config=config.execution,
    )
