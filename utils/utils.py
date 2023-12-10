import glob
import logging

import torch
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def cycle(dl):
    while True:
        for dt in dl:
            yield dt


def dynamic_batching(batch):
    src, tgt = zip(*batch)
    # Pad sequences to the maximum length
    src = pad_sequence(src, batch_first=True, padding_value=0)
    tgt = torch.tensor(tgt, dtype=torch.long)

    return src, tgt


def load_data(dataset_path):
    stories = []

    for story_path in tqdm(glob.glob(dataset_path + '/*')):
        with open(story_path, 'r') as f:
            story = f.read()
            stories.append(story)

    return stories


def train_tokenizer(vocab_size, stories, output_directory):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Metaspace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[START]", "[END]", "[MASK]"]
    )  # PAD index must be 0, it's used as default value in dynamic_batching()
    tokenizer.train_from_iterator(stories, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[START] $A [END]",
        special_tokens=[
            ("[START]", tokenizer.token_to_id("[START]")),
            ("[END]", tokenizer.token_to_id("[END]")),
        ]
    )
    tokenizer.decoder = decoders.Metaspace()

    tokenizer.save(str(output_directory / 'tokenizer.json'))
    logging.info(f"Tokenizer saved to: {output_directory / 'tokenizer.json'}")

    return tokenizer


def top_k_top_p_filtering(
        logits: Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
) -> Tensor:

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_logits.softmax(dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits
