import json
import torch
import os

import pickle
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence
 

def load_public_data(data_dir, data_name):
    with open(os.path.join(data_dir, data_name, "train.pkl"), "rb") as f:
        train = pickle.load(f)
    with open(os.path.join(data_dir, data_name, "dev.pkl"), "rb") as f:
        val = pickle.load(f)
    with open(os.path.join(data_dir, data_name, "test.pkl"), "rb") as f:
        test = pickle.load(f)

    item_size = train["dim_process"]
    padding_idx = item_size

    train_dataset = [{
        'item': torch.tensor([elem['type_event'] for elem in seq]),
        'time_stamp': torch.tensor([elem['time_since_start'] for elem in seq]),
        'time_gap': torch.tensor([elem['time_since_last_event'] for elem in seq])
    } for seq in train['train'] if len(seq) > 0]

    val_dataset = [{
        'item': torch.tensor([elem['type_event'] for elem in seq]),
        'time_stamp': torch.tensor([elem['time_since_start'] for elem in seq]),
        'time_gap': torch.tensor([elem['time_since_last_event'] for elem in seq])
    } for seq in val['dev'] if len(seq) > 0]

    test_dataset = [{
        'item': torch.tensor([elem['type_event'] for elem in seq]),
        'time_stamp': torch.tensor([elem['time_since_start'] for elem in seq]),
        'time_gap': torch.tensor([elem['time_since_last_event'] for elem in seq])
    } for seq in test['test'] if len(seq) > 0]

    total_len = [len(seq['item']) for seq in train_dataset] + [len(seq['item']) for seq in val_dataset] + [len(seq['item']) for seq in test_dataset]
    min_len = min(total_len)
    max_len = max(total_len)

    return train_dataset, val_dataset, test_dataset, item_size, padding_idx, min_len, max_len



def load_train_val_test(data_dir, data_name, seed = 2025, batch_size = 8, block_size = 512):
    torch.manual_seed(seed)
    
    train_dataset, val_dataset, test_dataset, item_size, padding_idx, min_len, max_len = load_public_data(data_dir, data_name)
    
    config = {
        'item_size': item_size,
        'padding_idx': padding_idx,
        'min_len': min_len,
        'max_len': max_len,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'batch_size': batch_size
    }

    generator = torch.Generator()
    generator.manual_seed(seed)

    def collate_fn(batch):
        items = [x['item'] for x in batch]
        time_stamps = [x['time_stamp'] for x in batch]
        time_gaps = [x['time_gap'] for x in batch]

        items_padded = pad_sequence(items, batch_first=True, padding_value=padding_idx)
        time_stamps_padded = pad_sequence(time_stamps, batch_first=True, padding_value=-1)
        time_gaps_padded = pad_sequence(time_gaps, batch_first=True, padding_value=-1)

        return {'item': items_padded, 'time_stamp': time_stamps_padded, 'time_gap': time_gaps_padded}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, generator=generator)

    return train_loader, val_loader, test_loader, config
