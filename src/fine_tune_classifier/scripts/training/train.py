#!/usr/bin/env python3

from transformers import Trainer, RobertaTokenizer
from datasets import load_from_disk, load_metric
from model_and_training_config import ModelAndConfig
import numpy as np
import logging
import sys
import argparse
import os
import tarfile
import boto3

# Load the accuracy metric
metric = load_metric("accuracy")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser()

    # Hyperparameters & Model Parameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--num_labels", type=int, default=6)

    # Directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "."))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--training_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "."))
    parser.add_argument("--test_dir", type=str, default=os.environ.get("SM_CHANNEL_TEST", "."))

    # Parse arguments
    args, _ = parser.parse_known_args()

    # Logging setup
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load datasets from disk
    try:
        train_dataset = load_from_disk(args.training_dir)
        test_dataset = load_from_disk(args.test_dir)
    except Exception as e:
        logging.error(f"Error loading datasets: {str(e)}")
        raise e

    # Import Model, Tokenizer, Training Args, and Data Collator 
    try:
        config = ModelAndConfig()
        model = config.model
        tokenizer = config.tokenizer
        training_args = config.training_args
        data_collator = config.data_collator
    except Exception as e:
        logging.error(f"Error initializing model and configurations: {str(e)}")
        raise e
    
    # Define the metrics computation function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    try:
        # Setup the Trainer instance and train the model
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        # Train and Evaluate
        trainer.train()
        eval_result = trainer.evaluate(eval_dataset=test_dataset)
    except Exception as e:
        logging.error(f"Error during training and evaluation: {str(e)}")
        raise e

    # Save the evaluation results to a file
    try:
        with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
            print(f"***** Eval results *****")
            for key, value in sorted(eval_result.items()):
                writer.write(f"{key} = {value}\n")
    except Exception as e:
        logging.error(f"Error saving evaluation results: {str(e)}")
        raise e

    # Save the trained model to disk
    try:
        trainer.save_model(args.model_dir)
    except Exception as e:
        logging.error(f"Error saving model: {str(e)}")
        raise e

    
    try:
        # Package the model
        model_archive = 'model.tar.gz'
        with tarfile.open(model_archive, 'w:gz') as archive:
            archive.add(args.model_dir, arcname=os.path.basename(args.model_dir))

        # Upload to S3
        s3_bucket = 'your-s3-bucket-name'  # Replace with your bucket name
        s3_prefix = 'models/'  # Optional: Path in your bucket where you want to store the model

        s3_client = boto3.client('s3')
        s3_client.upload_file(model_archive, s3_bucket, os.path.join(s3_prefix, model_archive))

        logger.info(f"Model saved to: s3://{s3_bucket}/{s3_prefix}{model_archive}")

    except Exception as e:
        logging.error(f"Error archiving and uploading model to S3: {str(e)}")
        raise e
