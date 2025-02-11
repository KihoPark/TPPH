import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from model.tpph import TPPH
from model.psdm import PSDM
from model.em import EM

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_load import load_train_val_test


argparser = argparse.ArgumentParser()
argparser.add_argument('--data_dir', type=str, required=True)
argparser.add_argument('--log_dir', type=str, required=True)
argparser.add_argument('--data_name', type=str, required = True)
argparser.add_argument('--num_components', type=int, default=16)
argparser.add_argument('--device', type=str, default='cuda')

argparser.add_argument('--model_name', type=str, default='TPPH', choices=['TPPH', 'PSDM'])

argparser.add_argument('--item_dim', type=int, default=256)
argparser.add_argument('--time_dim', type=int, default=256)
argparser.add_argument('--num_heads', type=int, default=4)
argparser.add_argument('--num_layers', type=int, default=1)

argparser.add_argument('--epochs', type=int, default=300)
argparser.add_argument('--lr', type=float, default=1e-3)
argparser.add_argument('--seed', type=int, default=2025)
argparser.add_argument('--batch_size', type=int, default=128)
argparser.add_argument('--block_size', type=int, default=512)

args = argparser.parse_args()


seed = args.seed
torch.manual_seed(seed)
device = torch.device(args.device)

train_loader, val_loader, test_loader, config = load_train_val_test(args.data_dir, args.data_name, seed = 2025, batch_size = args.batch_size)
item_size = config['item_size']
padding_idx = config['padding_idx']

if args.model_name == 'TPPH':
    model = TPPH(item_size=item_size, item_dim=args.item_dim, time_dim = args.time_dim, 
                  num_heads = args.num_heads, num_layers = args.num_layers,
                  num_components = args.num_components, padding_idx=padding_idx)
elif args.model_name == 'PSDM':
    model = PSDM(item_size=item_size, item_dim=args.item_dim, time_dim = args.time_dim, num_layers = args.num_layers,
                  num_components = args.num_components, padding_idx=padding_idx)
model = model.to(device)


print(f"Dataset {args.data_name}, Model {args.model_name}, Mixtures {model.num_components}, Layers {args.num_layers}, Seed {args.seed}")


## Initialization
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

means, variances, weights = EM(train_data, model.num_components, seed=args.seed)

## Train
with torch.no_grad():
    model.mu.copy_(means)
    model.log_sigma.copy_(variances.clamp(min=1e-2).log()/2)

os.makedirs(f'{args.log_dir}/{args.data_name}/{args.model_name}', exist_ok=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

store_val = torch.inf
log_list = {
    "config": {"data_name": args.data_name, "model_name": args.model_name, "num_components": model.num_components, "num_layers": args.num_layers, "seed": args.seed,
               "item_dim": args.item_dim, "time_dim": args.time_dim, "num_heads": args.num_heads, "lr": args.lr, "batch_size": args.batch_size, "block_size": args.block_size},
    "train": {"full_log_lik": [], "time_log_lik": [], "mark_loss": []},
    "val": {"full_log_lik": [], "time_log_lik": [], "mark_loss": []},
    "test": {"full_log_lik": [], "time_log_lik": [], "mark_loss": []},
    "computation_time": []
}
store_probs = []
store_time_gaps_list = []

best_epoch = 0

for epoch in range(args.epochs):
    model.train()
    train_time_loss = 0
    train_mark_loss = 0
    train_num_tokens = 0

    start_time = time.time()

    for batch in train_loader:
        items = batch['item'].to(device)
        time_stamps = batch['time_stamp'].to(device)
        time_gaps = batch['time_gap'].to(device)

        loss, probs, mu, sigma, time_loss, mark_loss = model.compute_loglik(items, time_stamps, time_gaps)
        with torch.no_grad():
            train_time_loss += time_loss
            train_mark_loss += mark_loss
            train_num_tokens += (items[:,1:] != padding_idx).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss = (train_time_loss + train_mark_loss)/train_num_tokens

    log_list['train']['full_log_lik'].append(-train_loss.item())
    log_list['train']['time_log_lik'].append((-train_time_loss.item() / train_num_tokens).cpu())
    log_list['train']['mark_loss'].append((train_mark_loss.item() / train_num_tokens).cpu())
    

    # Validation loop
    model.eval()
    val_time_loss = 0
    val_mark_loss = 0
    val_num_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            items = batch['item'].to(device)
            time_stamps = batch['time_stamp'].to(device)
            time_gaps = batch['time_gap'].to(device)

            loss, probs, mu, sigma, time_loss, mark_loss = model.compute_loglik(items, time_stamps, time_gaps)
            val_time_loss += time_loss
            val_mark_loss += mark_loss
            val_num_tokens += (items[:,1:] != padding_idx).sum()
        
        val_loss = (val_time_loss + val_mark_loss)/val_num_tokens

    if val_loss < store_val:
        best_epoch = epoch
        store_val = val_loss
        torch.save(model.state_dict(), f'{args.log_dir}/{args.data_name}/{args.model_name}/{model.num_components}_mix_{args.num_layers}_layers_{args.seed}.pth')
    
    log_list['val']['full_log_lik'].append(-val_loss.item())
    log_list['val']['time_log_lik'].append((-val_time_loss.item() / val_num_tokens).cpu())
    log_list['val']['mark_loss'].append((val_mark_loss.item() / val_num_tokens).cpu())
    
    if epoch % 30 == 2:
        print(f"Epoch {epoch}: Train: {-train_loss.item():.6f}, Val: {-val_loss.item():.6f}")
        print(f"Best Val: {-store_val.item():.6f} at epoch {best_epoch}")
        print(f"\t Epoch time: {epoch_time:.2f}")

    epoch_time = time.time() - start_time
    log_list['computation_time'].append(epoch_time)
    

## Test set
model.load_state_dict(torch.load(f'{args.log_dir}/{args.data_name}/{args.model_name}/{model.num_components}_mix_{args.num_layers}_layers_{args.seed}.pth'))
model.eval()
start_time = time.time()
test_time_loss = 0
test_mark_loss = 0
num_tokens = 0
with torch.no_grad():
    for batch in test_loader:
        items = batch['item'].to(device)
        time_stamps = batch['time_stamp'].to(device)
        time_gaps = batch['time_gap'].to(device)

        loss, probs, mu, sigma, time_loss, mark_loss = model.compute_loglik(items, time_stamps, time_gaps)
        test_time_loss += time_loss
        test_mark_loss += mark_loss
        num_tokens += (items[:,1:] != padding_idx).sum()
    
    test_loss = (test_time_loss + test_mark_loss)/num_tokens

print(f"Test log-likelihood: {-test_loss.item():.6f}")
log_list['test']['full_log_lik'] = -test_loss.item()
log_list['test']['time_log_lik'] = (-test_time_loss.item() / num_tokens).cpu()
log_list['test']['mark_loss'] = (test_mark_loss.item() / num_tokens).cpu()
epoch_time = time.time() - start_time
log_list['computation_time'].append(epoch_time)

torch.save(log_list, f'{args.log_dir}/{args.data_name}/{args.model_name}/{model.num_components}_mix_{args.num_layers}_layers_{args.seed}_log_list.pt')