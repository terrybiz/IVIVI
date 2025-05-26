from __future__ import print_function
import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

# clear those instances that have no positive instances to avoid training error
import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss_HNM(nn.Module):
    def __init__(self, temperature=0.07, hard_negative_ratio=0.5):
        super(SupConLoss_HNM, self).__init__()
        self.temperature = temperature
        self.hard_negative_ratio = hard_negative_ratio

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # positive mask

        # 相似度矩陣
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 避免自己對自己
        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        negative_mask = (1 - mask) * logits_mask  # 1 表示 negative

        hard_negatives_mask = torch.zeros_like(negative_mask)

        if self.hard_negative_ratio > 0.0:
            # 擷取所有負樣本的 logits 值
            neg_logits_all = logits[negative_mask.bool()]  # 1D tensor
            num_total_negatives = neg_logits_all.numel()
            num_hard_negatives = max(1, int(self.hard_negative_ratio * num_total_negatives))

            # 取 top-k 難的 logits 值作為門檻
            if num_total_negatives > 0:
                threshold = torch.topk(neg_logits_all, num_hard_negatives, largest=True).values[-1]

                # 標記出 hard negatives：logits 值 >= 門檻
                hard_negatives_mask = ((logits >= threshold) & (negative_mask.bool())).float()

        # 更新最終使用的 mask：正樣本 + hard negatives
        final_mask = mask + hard_negatives_mask
        final_mask = torch.clamp(final_mask, max=1.0) * logits_mask

        # 處理無正樣本情況
        single_samples = (final_mask.sum(1) == 0).float()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over selected positives
        mean_log_prob_pos = (final_mask * log_prob).sum(1) / (final_mask.sum(1) + single_samples)

        # compute loss
        loss = -mean_log_prob_pos * (1 - single_samples)
        loss = loss.sum() / (loss.shape[0] - single_samples.sum())

        return loss, hard_negatives_mask, final_mask

