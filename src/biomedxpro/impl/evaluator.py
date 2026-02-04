# src/biomedxpro/impl/evaluator.py
from typing import Sequence

import torch
from loguru import logger
from open_clip import create_model_from_pretrained, get_tokenizer

from biomedxpro.core.domain import (
    EncodedDataset,
    EvaluationMetrics,
    Individual,
    PromptEnsemble,
)
from biomedxpro.core.interfaces import IFitnessEvaluator
from biomedxpro.utils.metrics import calculate_classification_metrics


class FitnessEvaluator(IFitnessEvaluator):
    """
    Evaluates individuals using a frozen BioMedCLIP model.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        context_length: int = 256,
        batch_size: int = 64,  # Controls Prompt batching (not images, they are pre-loaded)
    ) -> None:
        self.device = torch.device(device)
        self.context_length = context_length
        self.batch_size = batch_size

        logger.info(f"Loading BioMedCLIP model: {model_name} on {self.device}")

        # Initialize OpenCLIP Model
        self.model, _ = create_model_from_pretrained(model_name=model_name)
        self.model.to(self.device).eval()
        self.tokenizer = get_tokenizer(model_name)

        # Cache logit scale
        self.logit_scale = self.model.logit_scale.exp().item()

    def compute_batch_probabilities(
        self,
        prompts: list[list[str]],
        dataset: EncodedDataset,
    ) -> torch.Tensor:
        """
        THE CORE PHYSICS ENGINE.
        Computes the class probability distribution for every sample-prompt pair.

        Args:
            prompts: List of prompt lists. Each inner list contains N_classes prompts.
                     e.g. [["class0_desc", "class1_desc", ...], ...]
            dataset: The dataset to evaluate on.

        Returns:
            torch.Tensor: Shape (N_samples, N_prompts, N_classes).
                          Values are probability distributions over classes.
        """
        image_feats = dataset.features.to(self.device).float()
        num_prompts = len(prompts)
        num_samples = dataset.num_samples

        # Detect number of classes from first prompt (assume uniform)
        if not prompts:
            raise ValueError("Cannot compute probabilities with empty prompt list")
        num_classes = len(prompts[0])

        # Validate all prompts have same class count
        for i, prompt_list in enumerate(prompts):
            if len(prompt_list) != num_classes:
                raise ValueError(
                    f"Prompt {i} has {len(prompt_list)} classes, expected {num_classes}"
                )

        # Output container: (N_samples, N_prompts, N_classes)
        all_probs = []

        # Process prompts in batches to avoid VRAM spikes with large populations
        for i in range(0, num_prompts, self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # 1. Flatten prompts: [[c0, c1, ...], ...] -> [c0, c1, ..., c0, c1, ...]
            flat_texts = []
            for prompt_list in batch_prompts:
                flat_texts.extend(prompt_list)

            # 2. Tokenize & Encode Text
            text_tokens = self.tokenizer(
                flat_texts, context_length=self.context_length
            ).to(self.device)

            is_cuda = self.device.type == "cuda"
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=is_cuda):
                text_feats = self.model.encode_text(text_tokens).float()
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                # Shape: (Embed_Dim, N_classes * Batch_Size)
                text_feats = text_feats.t()

                # 3. Compute Logits
                # (N_samples, Embed) @ (Embed, N_classes*Batch) -> (N_samples, N_classes*Batch)
                logits = self.logit_scale * (image_feats @ text_feats)

                # 4. Softmax over classes
                # Reshape to (N_samples, Batch_Size, N_classes)
                reshaped = logits.view(num_samples, len(batch_prompts), num_classes)
                probs = reshaped.softmax(dim=-1)

                # Keep full probability distribution (N_samples, Batch_Size, N_classes)
                all_probs.append(probs)

        # Concatenate all batches along the prompt dimension (dim 1)
        # Final Shape: (N_samples, N_prompts, N_classes)
        return torch.cat(all_probs, dim=1)

    def evaluate(
        self, individuals: Sequence[Individual], dataset: EncodedDataset
    ) -> None:
        """
        Batched evaluation for Evolution.
        Acts as a consumer of compute_batch_probabilities.
        """
        if not individuals:
            return

        # 1. Extract Prompts
        prompts = [list(ind.genotype.prompts) for ind in individuals]

        # 2. Get the Physics (Reuse the core logic!)
        # Shape: (N_samples, N_individuals, N_classes)
        probs_matrix = self.compute_batch_probabilities(prompts, dataset)

        # 3. Calculate Metrics (CPU-bound loop)
        y_true = dataset.labels.cpu().numpy()

        # Move probs to CPU once to avoid synchronization overhead in the loop
        probs_matrix_cpu = probs_matrix.cpu().float().numpy()

        for i, ind in enumerate(individuals):
            # Extract probability distribution for this individual: (N_samples, N_classes)
            y_prob_dist = probs_matrix_cpu[:, i, :]

            # For multi-class: use argmax instead of threshold
            y_pred = y_prob_dist.argmax(axis=1)

            # For metrics calculation, we need the probability of the predicted class
            # This is used for metrics like AUC in multi-class settings
            y_prob = y_prob_dist.max(axis=1)

            metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
            ind.update_metrics(metrics)

    def evaluate_ensemble(
        self,
        ensemble: PromptEnsemble,
        dataset: EncodedDataset,
    ) -> EvaluationMetrics:
        """
        Orchestrates the evaluation of a Deployment Model.

        Flow:
        1. Evaluator (Infrastructure) -> Computes raw probabilities (GPU).
        2. Ensemble (Domain) -> Applies voting weights (Pure Logic).
        3. Metrics (Utils) -> Calculates final report.
        """
        logger.info("Computing expert opinions on validation set...")

        # 1. GPU Compute
        # Shape: (N_samples, N_experts, N_classes)
        raw_probs = self.compute_batch_probabilities(ensemble.prompts, dataset)

        # 2. Domain Logic
        logger.info("Aggregating votes...")
        # Shape: (N_samples, N_classes)
        final_probs = ensemble.apply(raw_probs)

        # 3. Metric Calculation
        y_prob_dist = final_probs.cpu().numpy()
        y_true = dataset.labels.cpu().numpy()

        # Multi-class: use argmax for predictions
        y_pred = y_prob_dist.argmax(axis=1)

        # For metrics, use the max probability (predicted class confidence)
        y_prob = y_prob_dist.max(axis=1)

        return calculate_classification_metrics(y_true, y_pred, y_prob)
