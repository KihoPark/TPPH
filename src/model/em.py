import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def EM(data, num_components=20, seed = 2024):
    torch.manual_seed(seed)
    device = data.device
    weights = torch.ones(num_components, device=device) / num_components
    means = torch.linspace(data.min(), data.max(), num_components, device=device)
    variances = torch.ones(num_components, device=device) * 0.1

    for _ in range(50): 
        log_prob = -0.5 * ((data - means.unsqueeze(0)) ** 2 / variances.unsqueeze(0) + torch.log(variances.unsqueeze(0) * 2 * torch.pi))
        log_prob += torch.log(weights.unsqueeze(0))
        responsibilities = F.softmax(log_prob, dim=-1)

        Nk = responsibilities.sum(dim=0)
        weights = Nk / len(data)
        means = (responsibilities * data).sum(dim=0) / Nk
        variances = ((responsibilities * (data - means.unsqueeze(0)) ** 2).sum(dim=0) / Nk).clamp(min=1e-3)
    
    return means, variances, weights


def Mixture(train_loader, val_loader, num_components=20, device = None, padding_idx = None, seed = 2024):
    train_data = []
    padding_flags = []
    for i in train_loader:
        train_data.append(i['time_gap'].flatten())
        padding_flags.append((i['item'] == padding_idx).flatten())
    train_data = torch.cat(train_data)
    padding_flags = torch.cat(padding_flags)
    train_data = train_data[~padding_flags]

    train_data = torch.log(train_data[train_data > 0]).reshape(-1, 1)
    if device is not None:
        train_data = train_data.to(device)
    
    means, variances, weights = EM(train_data, num_components, seed)

    val_data = []
    padding_flags = []
    for i in val_loader:
        val_data.append(i['time_gap'].flatten())
        padding_flags.append((i['item'] == padding_idx).flatten())
    val_data = torch.cat(val_data)
    padding_flags = torch.cat(padding_flags)
    val_data = val_data[~padding_flags]
    
    val_data = torch.log(val_data[val_data > 0]).reshape(-1, 1)
    val_data = val_data.to(device)

    log_prob_validation = -0.5 * ((val_data - means.unsqueeze(0)) ** 2 / variances.unsqueeze(0) + torch.log(variances.unsqueeze(0) * 2 * torch.pi))
    log_prob_validation += torch.log(weights)
    log_prob_validation = torch.logsumexp(log_prob_validation, dim=-1)
    total_log_likelihood_validation = log_prob_validation.sum()

    log_lik = total_log_likelihood_validation - val_data.sum()
    val_log_lik = log_lik.item() / len(val_data)
    return means, variances, weights, val_log_lik