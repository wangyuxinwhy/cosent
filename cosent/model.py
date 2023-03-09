from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

# https://kexue.fm/archives/8847
class CosineSimilarityLoss(nn.Module):
    bias: torch.Tensor

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.register_buffer('bias', torch.tensor([0.0]))

    def forward(self, predict_similarity: torch.Tensor, true_similarity: torch.Tensor) -> torch.Tensor:
        # cosine_similarity: [batch_size], true_similarity: [batch_size]
        predict_similarity = predict_similarity * self.alpha

        cosine_similarity_diff = -(predict_similarity.unsqueeze(0) - predict_similarity.unsqueeze(1))
        smaller_mask = true_similarity.unsqueeze(0) <= true_similarity.unsqueeze(1)
        cosine_similarity_diff = cosine_similarity_diff.masked_fill(smaller_mask, -1e12)

        cosine_diff_scores_add_bias = torch.cat((cosine_similarity_diff.view(-1), self.bias))

        loss = torch.logsumexp(cosine_diff_scores_add_bias, dim=0)
        return loss


# https://arxiv.org/pdf/2203.02155.pdf
class SoftRankLoss(nn.Module):
    bias: torch.Tensor

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.register_buffer('bias', torch.tensor([0.0]))

    def forward(self, predict_scores: torch.Tensor, true_scores: torch.Tensor) -> torch.Tensor:
        batch_size = predict_scores.size(0)

        predict_scores = predict_scores * self.alpha
        scores_diff = -(predict_scores.unsqueeze(0) - predict_scores.unsqueeze(1))
        smaller_mask = true_scores.unsqueeze(0) <= true_scores.unsqueeze(1)
        num_not_mask_count = batch_size * (batch_size-1)/2
        scores_diff = scores_diff.masked_fill(smaller_mask, -1e12).view(-1)
        loss = -torch.log(torch.sigmoid(-scores_diff)).sum() / num_not_mask_count
        return loss


class CoSentModel(nn.Module):
    def __init__(self, model_name_or_path: str, alpha: float = 20):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        # self.criterion = CosineSimilarityLoss(alpha)
        self.criterion = SoftRankLoss(alpha)

    def forward(self, text_ids: torch.Tensor, text_pair_ids: torch.Tensor, similarities: torch.Tensor | None = None):
        text_cls_embeddings = self.encoder(text_ids).last_hidden_state[:, 0, :]
        text_pair_cls_embeddings = self.encoder(text_pair_ids).last_hidden_state[:, 0, :]

        pred_similarities = torch.cosine_similarity(text_cls_embeddings, text_pair_cls_embeddings, dim=1)
        output = {'similarities': pred_similarities}

        if similarities is not None:
            loss = self.criterion(pred_similarities, similarities)
            output['loss'] = loss

        return output
