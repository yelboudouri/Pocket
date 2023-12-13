# Pocket: a Transformer-Based Language Model Tailored for Short Story Generation

Pocket is a sophisticated language model designed specifically for the creative task of short story generation. Leveraging state-of-the-art Transformer architecture, Pocket is capable of crafting engaging narratives from diverse datasets of short stories. This repository provides a comprehensive set of tools for training the model on custom datasets and generating captivating stories.

## Getting Started

### Training

#### 1. Prepare Your Data

Organize your short stories dataset in the data folder. Each story should be a separate text file.

#### 2. Install dependencies

    pip install -r requirements.txt

#### 3. Train the Model

    > python train.py -h
    usage: train.py [-h] [--dataset-path DATASET_PATH] [--output-dir OUTPUT_DIR] [--train-steps TRAIN_STEPS] [--eval-steps EVAL_STEPS] [--use-amp]
                    [--batch-size BATCH_SIZE] [--lr LR] [--checkpoint CHECKPOINT]
    
    Train Pocket: a casual language model
    
    optional arguments:
      -h, --help            show this help message and exit
      --dataset-path DATASET_PATH
                            Path to the directory containing text files, each file representing a story.
      --output-dir OUTPUT_DIR
                            Path where the best model will be saved. Defaults to 'out'.
      --train-steps TRAIN_STEPS
                            Number of training steps.
      --eval-steps EVAL_STEPS
                            Number of training steps.
      --use-amp             Enable mixed precision training for faster computation.
      --batch-size BATCH_SIZE
                            Batch size for training.
      --lr LR               Learning rate for the optimizer.
      --checkpoint CHECKPOINT
                            Path to a checkpoint file for resuming or fine-tuning training.


### Generation

    > python generate.py -h
    usage: generate.py [-h] [--model-artifacts MODEL_ARTIFACTS] [--temperature TEMPERATURE] [--top-k TOP_K] [--top-p TOP_P]
    
    Generate a story
    
    optional arguments:
      -h, --help            show this help message and exit
      --model-artifacts MODEL_ARTIFACTS
                            Directory containing 'tokenizer.json' and 'last_model.pt'.
      --temperature TEMPERATURE
                            Temperature parameter for controlling the randomness of text generation
      --top-k TOP_K         Top-k parameter for controlling the diversity of text generation
      --top-p TOP_P         Top-p parameter for controlling the diversity of text generation


## Contribution

We welcome contributions from the community! Whether you find a bug, want to add a feature, or improve documentation, feel free to submit a pull request.