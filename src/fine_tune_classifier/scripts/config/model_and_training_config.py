#!/usr/bin/env python3

# Importing necessary classes and functions from the transformers library
from transformers import (
    RobertaForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    RobertaTokenizer,
)

# Specify the path to the DeepSpeed configuration file.
# This is set to look in the current directory by default.
DEEPSPEED_CONFIG_PATH = "./ds_config.json"


class ModelAndConfig:
    """
    This class encapsulates the tokenizer, model, training arguments, and data collator
    needed for training and inference.

    Attributes:
    - tokenizer: Tokenizer used for preprocessing text data.
    - model: The pre-trained model architecture used for classification.
    - training_args: Arguments specifying training configuration and parameters.
    - data_collator: Collates batches of data with padding.
    """

    def __init__(self):
        # Initialize tokenizer with "roberta-base" model.
        # Used to convert text data into a format suitable for the model.
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Initialize model with "roberta-base" architecture.
        self.model = RobertaForSequenceClassification.from_pretrained("roberta-base")

        # Define training arguments.
        # These specify various training parameters and configurations.
        self.training_args = TrainingArguments(
            output_dir="./results",  # Directory where training outputs will be saved.
            num_train_epochs=3,  # Number of training epochs.
            per_device_train_batch_size=16,  # Training batch size per device (GPU/CPU).
            per_device_eval_batch_size=64,  # Evaluation batch size per device (GPU/CPU).
            evaluation_strategy="epoch",  # Evaluation and logging will be done at the end of each epoch.
            logging_dir="./logs",  # Directory where logs will be saved.
            deepspeed=DEEPSPEED_CONFIG_PATH,  # Path to the DeepSpeed config file.
        )

        # Initializing data collator.
        # Will be used to collate batches of data with padding during training.
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
