import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttn(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

        attn_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        attention_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.out(attention_output)
        
        return output, attn_weights


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim, dropout=0.2):
        super().__init__()
        self.self_attn = SelfAttn(embed_dim=hidden_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, mask)
        x = x + self.norm1(attn_output)
        
        mlp_output = self.mlp(x)
        x = x + self.norm2(mlp_output)
        
        return x
    

class TPPH(nn.Module):
    def __init__(self, item_size, item_dim, time_dim, num_heads, num_layers, num_components, padding_idx):
        super().__init__()
        self.item_dim = item_dim
        self.time_dim = time_dim
        self.hidden_dim = item_dim + time_dim
        self.num_heads = num_heads
        self.mlp_dim = 2 * self.hidden_dim
        self.padding_idx = padding_idx
        self.num_components = num_components
        self.embedding = nn.Embedding(item_size+1, item_dim, padding_idx=self.padding_idx)
        self.div_term = torch.exp(torch.arange(0, time_dim, 2) * -(6 * math.log(10.0) / time_dim)).reshape(1, 1, -1)

        self.self_attn = SelfAttn(embed_dim=self.hidden_dim, num_heads=self.num_heads)

        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout=0.2
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
    
    def create_autoregressive_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, seq_len, seq_len)

    def forward(self, items, time_stamps):
        padding_mask = (items == self.padding_idx)
        item_enc = self.embedding(items)
        item_enc = F.normalize(item_enc, p=2, dim=-1, eps=1e-16)
        
        tem_enc = self.compute_temporal_embedding(time_stamps)
        tem_enc = tem_enc.masked_fill(padding_mask.unsqueeze(-1), 0)
        
        x = torch.cat([item_enc, tem_enc], dim=-1)
        
        mask = self.create_autoregressive_mask(x.size(1)).to(x.device)  # (1, 1, seq_len, seq_len)
        batch_size, seq_len = x.size(0), x.size(1)
        mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
        
        for layer in self.layers:
            x = layer(x, mask)

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
        

        # mark prediction
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

