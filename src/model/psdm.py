import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PSDM(nn.Module):
    def __init__(self, item_size, item_dim, time_dim, num_layers, num_components, padding_idx):
        super().__init__()
        self.item_dim = item_dim
        self.time_dim = time_dim
        self.hidden_dim = item_dim + time_dim
        self.mlp_dim = 2 * self.hidden_dim
        self.padding_idx = padding_idx
        self.num_components = num_components
        self.embedding = nn.Embedding(item_size+1, item_dim, padding_idx=self.padding_idx)
        self.div_term = torch.exp(torch.arange(0, time_dim, 2) * -(6 * math.log(10.0) / time_dim)).reshape(1, 1, -1)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.mlp_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(self.mlp_dim, self.hidden_dim),
            ) for _ in range(num_layers)
        ])

        self.current_mlp = nn.Sequential(
            nn.Linear(self.time_dim, 2* self.time_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2* self.time_dim, self.hidden_dim),
        )

        self.current_mlp2 = nn.Sequential(
            nn.Linear(self.hidden_dim, 2* self.time_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2* self.time_dim, self.item_dim),
        )

        self.mu = nn.Parameter(torch.randn(self.num_components))
        self.log_sigma = nn.Parameter(torch.randn(self.num_components))
        self.type_predictor = nn.Linear(self.hidden_dim, self.num_components, bias=False)

    def compute_temporal_embedding(self, time):
        batch_size = time.size(0)
        seq_len = time.size(1)
        te = torch.zeros(batch_size, seq_len, self.time_dim).to(time.device)
        _time = time.unsqueeze(-1)
        div_term = self.div_term.to(time.device)
        te[..., 0::2] = torch.sin(_time * div_term)
        te[..., 1::2] = torch.cos(_time * div_term)
        return te
    
    def forward(self, items, time_stamps):
        padding_mask = (items == self.padding_idx)
        item_enc = self.embedding(items)
        item_enc = F.normalize(item_enc, p=2, dim=-1, eps=1e-16)

        tem_enc = self.compute_temporal_embedding(time_stamps)
        tem_enc = tem_enc.masked_fill(padding_mask.unsqueeze(-1), 0)

        x = torch.cat([item_enc, tem_enc], dim=-1)
        for layer in self.layers:
            x = layer(x)
        x = F.normalize(x, p=2, dim=-1, eps=1e-16)

        return x

    def compute_loglik(self, items, time_stamps, time_gaps):
        embed = self.forward(items[:,:-1], time_stamps[:,:-1])
        logits = self.type_predictor(embed)
        probs = F.softmax(logits, dim=-1)

        time_gaps = time_gaps[:,1:]
        time_gaps_mask = (items[:,1:] == self.padding_idx)
        time_gaps_masked = torch.where(items[:,1:] == self.padding_idx, torch.tensor(1e-16, device=time_gaps.device), time_gaps)
        time_gaps_masked[time_gaps_masked == 0] = 1e-16
        log_delta = torch.log(time_gaps_masked)

        mu = self.mu
        sigma = torch.exp(self.log_sigma)
        sigma_sq = sigma ** 2

        constant = torch.log(torch.sqrt(torch.tensor(2 * torch.pi)))

        log_den = -0.5 * ((log_delta.unsqueeze(-1) - mu) ** 2) / sigma_sq - self.log_sigma - constant
        log_weighted_den = (torch.log(probs) + log_den)
        log_weighted_den[time_gaps_mask, :] = -float('inf')

        sum_log_weighted_den = torch.logsumexp(log_weighted_den, dim=-1)
        log_likelihoods = sum_log_weighted_den[~time_gaps_mask].sum() - log_delta[~time_gaps_mask].sum()
        time_loss = -log_likelihoods

        current_time = time_stamps[:,1:]
        tem_enc = self.compute_temporal_embedding(current_time)
        tem_enc = self.current_mlp(tem_enc)
        mark_embed = embed + tem_enc
        mark_embed = self.current_mlp2(mark_embed)

        current_item = items[:,1:]
        current_item = current_item.long()

        logits = torch.einsum('bik,vk->biv', mark_embed, self.embedding.weight)

        mark_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), current_item.reshape(-1), ignore_index=self.padding_idx, reduction='sum')

        loss = time_loss + mark_loss

        return loss, probs, mu, sigma, time_loss, mark_loss
