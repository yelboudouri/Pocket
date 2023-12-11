import argparse
import logging
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.config import PocketConfig
from model.pocket import Pocket
from utils.utils import top_k_top_p_filtering


logging.root.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s: %(message)s')


class Generator:

    def __init__(self,
                 model,
                 device,
                 max_length,
                 temperature,
                 top_k,
                 top_p):
        self.model = model
        self.device = device
        self.max_length = max_length

        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        self.start_token = tokenizer.token_to_id("[START]")
        self.end_token = tokenizer.token_to_id("[END]")
        self.pad_token = tokenizer.token_to_id("[PAD]")
        self.mask_token = tokenizer.token_to_id("[MASK]")

    def sample(self):
        generated_ids = torch.full((1, self.max_length), self.pad_token, device=self.device)
        generated_ids[0][0] = self.start_token

        token_index = 1
        while True:
            with torch.no_grad():
                generated_ids[0][token_index] = self.mask_token
                logits = self.model(generated_ids)
                logits /= self.temperature
                filtered_logits = top_k_top_p_filtering(logits, self.top_k, self.top_p)
                # Sample from the probability distribution
                predicted_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), 1).squeeze()
                generated_ids[0][token_index] = predicted_token
            token_index += 1

            if predicted_token == self.end_token or token_index == self.max_length:
                break

        generated_ids = generated_ids.cpu().numpy()[0]
        return generated_ids


def get_args():
    parser = argparse.ArgumentParser(description="Generate a story")

    parser.add_argument("--model-artifacts", type=str, default="out",
                        help="Directory containing 'tokenizer.json' and 'last_model.pt'.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature parameter for controlling the randomness of text generation")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k parameter for controlling the diversity of text generation")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p parameter for controlling the diversity of text generation")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    tokenizer = Tokenizer.from_file(str(Path(args.model_artifacts) / 'tokenizer.json'))
    logging.info(f"Loaded tokenizer from: {Path(args.model_artifacts) / 'tokenizer.json'}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device used: {device}")

    data = torch.load(str(Path(args.model_artifacts) / 'last_model.pt'), map_location=device)

    model_config = PocketConfig()
    model_config.pad_id = tokenizer.token_to_id("[PAD]")
    model_config.mask_id = tokenizer.token_to_id("[MASK]")

    model = Pocket(
        config=model_config,
        device=device
    )
    model.eval()
    model.load_state_dict(data['model'])

    ids = Generator(
        model,
        device,
        model_config.n_positions,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    ).sample()

    print(f"Ids: {ids}")
    print(f"Decoded: {tokenizer.decode(ids)}")
