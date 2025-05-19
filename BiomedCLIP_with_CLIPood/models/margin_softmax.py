import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginMetricSoftmax(nn.Module):
    def __init__(self, margins):
        super().__init__()
        self.margins = margins  # Tensor [num_classes]

    def forward(self, logits, labels):
        margins = self.margins.to(logits.device)
        one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        adjusted_logits = logits - one_hot * margins
        return F.cross_entropy(adjusted_logits, labels)

def compute_class_margins(text_embeddings, alpha=0.1):
    sim = F.cosine_similarity(text_embeddings.unsqueeze(1), text_embeddings.unsqueeze(0), dim=-1)
    sim.fill_diagonal_(0)
    avg_sim = sim.mean(dim=1)
    margins = alpha * avg_sim
    return margins
