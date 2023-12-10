import random

import numpy as np
import torch
from torch.utils.data import Dataset


class StoriesDataset(Dataset):

    def __init__(self, data, tokenizer, max_length):
        self.tokenized_data = tokenizer.encode_batch(data)

        # Filter sequences longer then max_length
        self.tokenized_data = [item for item in self.tokenized_data if len(item.ids) <= max_length]

        self.end_token = tokenizer.token_to_id("[END]")
        self.mask_token = tokenizer.token_to_id("[MASK]")

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, index):
        story = self.tokenized_data[index]

        ids = story.ids

        mask_index = random.randint(1, ids.index(self.end_token))
        target_id = ids[mask_index]

        input_ids = np.array(ids)
        input_ids = input_ids[:mask_index]
        input_ids = np.append(input_ids, self.mask_token)

        return torch.tensor(input_ids), target_id
