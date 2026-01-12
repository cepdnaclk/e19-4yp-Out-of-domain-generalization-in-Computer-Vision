from typing import Sequence, Tuple

import torch
from loguru import logger
from open_clip import create_model_from_pretrained, get_tokenizer

from biomedxpro.core.domain import EncodedDataset, Individual
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
        prompts: list[Tuple[str, str]],
        dataset: EncodedDataset,
    ) -> torch.Tensor:
        """
        THE CORE PHYSICS ENGINE.
        Computes the positive class probability for every sample-prompt pair.

        Args:
            prompts: List of (negative_prompt, positive_prompt).
            dataset: The dataset to evaluate on.

        Returns:
            torch.Tensor: Shape (N_samples, N_prompts).
                          Values are probabilities [0.0, 1.0] for the positive class.
        """
        image_feats = dataset.features.to(self.device).float()
        num_prompts = len(prompts)
        num_samples = dataset.num_samples

        # Output container: (N_samples, N_prompts)
        # We pre-allocate on CPU or GPU depending on memory constraints.
        # For now, let's keep on GPU for speed, move to CPU if needed.
        all_probs = []

        # Process prompts in batches to avoid VRAM spikes with large populations
        for i in range(0, num_prompts, self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]

            # 1. Flatten prompts: [(Neg1, Pos1), ...] -> [Neg1, Pos1, ...]
            flat_texts = []
            for neg, pos in batch_prompts:
                flat_texts.append(neg)
                flat_texts.append(pos)

            # 2. Tokenize & Encode Text
            text_tokens = self.tokenizer(
                flat_texts, context_length=self.context_length
            ).to(self.device)

            is_cuda = self.device.type == "cuda"
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=is_cuda):
                text_feats = self.model.encode_text(text_tokens).float()
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                # Shape: (Embed_Dim, 2 * Batch_Size)
                text_feats = text_feats.t()

                # 3. Compute Logits
                # (N_samples, Embed) @ (Embed, 2*Batch) -> (N_samples, 2*Batch)
                logits = self.logit_scale * (image_feats @ text_feats)

                # 4. Softmax
                # Reshape to (N_samples, Batch_Size, 2)
                # Columns: [Neg, Pos]
                reshaped = logits.view(num_samples, len(batch_prompts), 2)
                probs = reshaped.softmax(dim=-1)

                # Extract Positive Class (Index 1) -> (N_samples, Batch_Size)
                all_probs.append(probs[:, :, 1])

        # Concatenate all batches along the prompt dimension (dim 1)
        # Final Shape: (N_samples, N_prompts)
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
        prompts = [
            (ind.genotype.negative_prompt, ind.genotype.positive_prompt)
            for ind in individuals
        ]

        # 2. Get the Physics (Reuse the core logic!)
        # Shape: (N_samples, N_individuals)
        probs_matrix = self.compute_batch_probabilities(prompts, dataset)

        # 3. Calculate Metrics (CPU-bound loop)
        y_true = dataset.labels.cpu().numpy()

        # Move probs to CPU once to avoid synchronization overhead in the loop
        probs_matrix_cpu = probs_matrix.cpu().float().numpy()

        for i, ind in enumerate(individuals):
            y_prob = probs_matrix_cpu[:, i]

            # Simple thresholding for metrics
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
            ind.update_metrics(metrics)
