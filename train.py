import argparse
import logging
import os
from multiprocessing import cpu_count
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from tqdm import tqdm

import wandb
from model.config import PocketConfig
from model.pocket import Pocket
from utils.dataset import StoriesDataset
from utils.scheduler import CosineAnnealingWithWarmRestartsLR
from utils.utils import load_data, train_tokenizer, cycle, dynamic_batching

logging.root.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s: %(message)s')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Trainer:

    def __init__(self,
                 model,
                 device,
                 dataset,
                 output_dir,
                 checkpoint=None,
                 training_steps: int = 20000,
                 evaluating_steps: int = 100,
                 batch_size: int = 256,
                 lr: float = 1e-4,
                 use_amp: bool = False):
        self.device = device
        self.use_amp = use_amp
        self.output_dir = output_dir
        self.training_steps = training_steps
        self.evaluating_steps = evaluating_steps

        # Split into train / validation partitions
        n_val = int(len(dataset) * 0.2)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        logging.info(f"Stories used for training: {n_train}")
        logging.info(f"Stories used for evaluation: {n_val}")

        if n_val < batch_size or n_train < batch_size:
            raise ValueError("Subsets (train and val) size must be bigger then batch_size.")

        self.train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                           num_workers=cpu_count(), drop_last=True, pin_memory=True,
                                           collate_fn=dynamic_batching)
        self.val_dataloader = DataLoader(val_set, batch_size=batch_size,
                                         num_workers=cpu_count(), drop_last=True, pin_memory=True,
                                         collate_fn=dynamic_batching)
        self.train_dataloader = cycle(self.train_dataloader)
        self.val_dataloader = cycle(self.val_dataloader)

        self.model = model

        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWithWarmRestartsLR(self.optimizer, warmup_steps=64, cycle_steps=512, max_lr=lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        if checkpoint is not None:
            self.load(checkpoint)

        wandb.init(project="Pocket", anonymous='must', resume="allow")

    def train(self):
        self.model.train()

        eval_loss = None
        perplexity = None

        pbar = tqdm(unit="batch", total=self.training_steps, desc="Training: ")
        for step in range(self.training_steps):
            inputs, targets = next(self.train_dataloader)

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.autocast(device_type=self.device, enabled=self.use_amp):
                _, loss = self.model(inputs, targets)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.scheduler.step()

            # Update progress bar
            pbar.set_postfix({
                'train_loss': loss.item(),
                'eval_loss': eval_loss,
                'perplexity': perplexity
            })
            pbar.update(1)

            if step % 50 == 0:
                eval_loss, perplexity = self.eval()
                wandb.log({
                    'Train Loss': loss.item(),
                    'Eval Loss': eval_loss,
                    'Perplexity': perplexity
                })
                self.save(step)

        pbar.close()
        logging.info("Training finished!")

    def eval(self):
        self.model.eval()
        total_loss = 0.0

        for step in range(self.evaluating_steps):
            inputs, targets = next(self.val_dataloader)

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                logits, loss = self.model(inputs, targets)

            total_loss += loss.item()

        average_loss = total_loss / self.evaluating_steps
        perplexity = torch.exp(torch.tensor(average_loss))

        self.model.train()
        return average_loss, perplexity.item()

    def save(self, step):
        dt = {
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            'step': step
        }

        torch.save(dt, str(self.output_dir / 'last_model.pt'))

    def load(self, checkpoint):
        logging.info(f"Loaded checkpoint from: {checkpoint}")
        dt = torch.load(args.checkpoint, map_location=self.device)

        self.model.load_state_dict(dt['model'])
        self.optimizer.load_state_dict(dt['opt'])
        self.scaler.load_state_dict(dt['scaler'])
        self.scheduler.load_state_dict(dt['scheduler'])


def get_args():
    parser = argparse.ArgumentParser(description="Train Pocket")

    parser.add_argument("--dataset-path", type=str,
                        help="Path to the directory containing text files, each file representing a story.")
    parser.add_argument("--output-dir", type=str, default="out",
                        help="Path where the best model will be saved. Defaults to 'out'.")
    parser.add_argument("--train-steps", type=int, default=20000, help="Number of training steps.")
    parser.add_argument("--eval-steps", type=int, default=500, help="Number of training steps.")
    parser.add_argument('--use-amp', action='store_true', default=False,
                        help='Enable mixed precision training for faster computation.')
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint file for resuming or fine-tuning training.')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    output_directory = Path(args.output_dir)
    output_directory.mkdir(exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device used: {device}")

    model_config = PocketConfig()

    data = load_data(args.dataset_path)
    if (output_directory / 'tokenizer.json').exists():
        tokenizer = Tokenizer.from_file(str(output_directory / 'tokenizer.json'))
        logging.info(f"Tokenizer loaded from: {output_directory / 'tokenizer.json'}")
    else:
        tokenizer = train_tokenizer(model_config.vocab_size, data, output_directory)

    ds = StoriesDataset(data, tokenizer, model_config.n_positions)

    model_config.pad_id = tokenizer.token_to_id("[PAD]")
    model_config.mask_id = tokenizer.token_to_id("[MASK]")

    model = Pocket(
        config=model_config,
        device=device
    )
    summary(model, (args.batch_size, model_config.n_positions + 1), dtypes=[torch.long])

    Trainer(
        model,
        device,
        ds,
        output_dir=output_directory,
        checkpoint=args.checkpoint,
        training_steps=args.train_steps,
        evaluating_steps=args.eval_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        use_amp=args.use_amp
    ).train()
