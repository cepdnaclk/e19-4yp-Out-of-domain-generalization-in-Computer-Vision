from typing import Sequence

import numpy as np
import torch
from loguru import logger
from open_clip import create_model_from_pretrained, get_tokenizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from biomedxpro.core.domain import EncodedDataset, EvaluationMetrics, Individual
from biomedxpro.core.interfaces import IFitnessEvaluator


class FitnessEvaluator(IFitnessEvaluator):
    """
    Evaluates individuals using a frozen BioMedCLIP model.
    Optimized for high throughput by pre-computing image embeddings
    and batch-encoding text prompts.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        context_length: int = 256,
        batch_size: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.context_length = context_length
        self.batch_size = batch_size

        logger.info(f"Loading BioMedCLIP model: {model_name} on {self.device}")

        # Initialize OpenCLIP Model
        # We don't need the preprocess transform for evaluation since we use pre-computed embeddings
        self.model, _ = create_model_from_pretrained(model_name=model_name)
        self.model.to(self.device).eval()
        self.tokenizer = get_tokenizer(model_name)

        # Cache logit scale for slightly faster access
        self.logit_scale = self.model.logit_scale.exp().item()

    def evaluate(
        self, individuals: Sequence[Individual], dataset: EncodedDataset
    ) -> None:
        """
        Batched evaluation of the population.
        Updates each Individual.metrics in-place.
        """
        if not individuals:
            return

        logger.debug(
            f"Evaluating {len(individuals)} individuals on {dataset.num_samples} samples."
        )

        # 1. Prepare Image Features and Labels
        image_feats = dataset.features.to(self.device).float()
        labels = dataset.labels.to(self.device)
        y_true = labels.cpu().numpy()

        # 2. Batch Tokenize All Prompts
        all_texts = []
        for ind in individuals:
            all_texts.append(ind.genotype.negative_prompt)
            all_texts.append(ind.genotype.positive_prompt)

        text_tokens = self.tokenizer(all_texts, context_length=self.context_length).to(
            self.device
        )

        # 3. Batch Encode Text (One Forward Pass)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_feats = self.model.encode_text(text_tokens).float()
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # 4. Compute Metrics in chunks of individuals to prevent OOM
        # Memory consumption is O(N_samples * batch_size) instead of O(N_samples * N_individuals)
        num_individuals = len(individuals)
        for start_idx in range(0, num_individuals, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_individuals)
            batch_count = end_idx - start_idx

            # Extract text features for this batch of individuals
            # Each individual has 2 prompts (neg, pos)
            # Shape: (batch_count * 2, Embed_Dim)
            batch_text_feats = text_feats[start_idx * 2 : end_idx * 2].t()

            with torch.no_grad():
                # Compute Logits for this batch
                # (N_samples, Embed_Dim) @ (Embed_Dim, 2 * batch_count) -> (N_samples, 2 * batch_count)
                batch_logits = self.logit_scale * (image_feats @ batch_text_feats)

            # Calculate and store metrics for each individual in the current batch
            for i in range(batch_count):
                global_ind_idx = start_idx + i
                ind = individuals[global_ind_idx]

                # Extract logits for specific individual
                individual_logits = batch_logits[:, 2 * i : 2 * i + 2]
                probs = individual_logits.softmax(dim=1)

                # Column 1 is the Positive Class Probability
                y_prob = probs[:, 1].cpu().float().numpy()
                y_pred = individual_logits.argmax(dim=1).cpu().numpy()

                metrics = self._calculate_metrics(y_true, y_pred, y_prob)
                ind.update_metrics(metrics)

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> EvaluationMetrics:
        """
        Computes standard classification metrics and the custom Inverted BCE fitness.
        """
        # 1. Inverted BCE (The Fitness Function)
        # We implement it manually or via torch to match your reference logic
        # Using numpy/scikit-learn for consistency
        # Avoid log(0) errors with clipping
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)

        # Manual BCE calculation matching F.binary_cross_entropy behavior
        # BCE = - (y * log(p) + (1-y) * log(1-p))
        bce_loss = -np.mean(
            y_true * np.log(y_prob_clipped) + (1 - y_true) * np.log(1 - y_prob_clipped)
        )
        inverted_bce = 1.0 / (1.0 + bce_loss)

        # 2. Standard Metrics
        return {
            "inverted_bce": float(inverted_bce),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auc": float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else 0.5,
            "f1_macro": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
