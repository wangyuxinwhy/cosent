from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class SimilarityLoss(nn.Module):
    bias: torch.Tensor

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.register_buffer('bias', torch.tensor([0.0]))

    def forward(self, pred_similarity: torch.Tensor, true_similarity: torch.Tensor) -> torch.Tensor:
        # cosine_similarity: [batch_size], true_similarity: [batch_size]
        pred_similarity = pred_similarity * self.alpha

        cosine_similarity_diff = -(pred_similarity.unsqueeze(0) - pred_similarity.unsqueeze(1))
        smaller_mask = true_similarity.unsqueeze(0) <= true_similarity.unsqueeze(1)
        cosine_similarity_diff = cosine_similarity_diff.masked_fill(smaller_mask, -1e12)

        cosine_diff_scores_add_bias = torch.cat((cosine_similarity_diff.view(-1), self.bias))

        loss = torch.logsumexp(cosine_diff_scores_add_bias, dim=0)
        return loss


class CoSentModel(nn.Module):
    def __init__(self, model_name_or_path: str, alpha: float = 20):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.criterion = SimilarityLoss(alpha)

    def forward(self, text_ids: torch.Tensor, text_pair_ids: torch.Tensor, similarities: torch.Tensor | None = None):
        text_cls_embeddings = self.encoder(text_ids).last_hidden_state[:, 0, :]
        text_pair_cls_embeddings = self.encoder(text_pair_ids).last_hidden_state[:, 0, :]

        pred_similarities = torch.cosine_similarity(text_cls_embeddings, text_pair_cls_embeddings, dim=1)
        output = {'similarities': pred_similarities}

        if similarities is not None:
            loss = self.criterion(pred_similarities, similarities)
            output['loss'] = loss

        return output
