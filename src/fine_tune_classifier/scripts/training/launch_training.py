#!/usr/bin/env python3

# Import necessary libraries
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role
import logging
import boto3
import argparse

# Initialize logging to capture information and errors
logging.basicConfig(level=logging.INFO)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Launch a SageMaker training job.")
parser.add_argument(
    "--mode",
    type=str,
    choices=["distributed", "single-instance"],
    required=True,
    help="Specify the mode: distributed for multi-instance training or single-instance for single-instance training.",
)
args = parser.parse_args()

# Initialize a SageMaker session and get the execution role for the current SageMaker environment
try:
    sagemaker_session = sagemaker.Session()
    role = get_execution_role()
except Exception as e:
    logging.error(f"Error initializing SageMaker: {str(e)}")
    raise e

# Define the HuggingFace estimator based on the chosen mode
try:
    if args.mode == "distributed":
        instance_type = "ml.p3.8xlarge"  # Instance type for distributed training. Feel free to choose from: https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html
        instance_count = 2  # Number of instances for distributed training
        distribution = {"smdistributed": {"dataparallel": {"enabled": True}}}
    else:
        instance_type = "ml.p3.2xlarge"  # Instance type for single-instance training
        instance_count = 1  # Only one instance for single-instance training
        distribution = None

    huggingface_estimator = HuggingFace(
        entry_point="train.py",  # The Python script for training
        source_dir=".",  # Source code directory
        role=role,  # SageMaker execution role
        instance_type=instance_type,
        instance_count=instance_count,
        transformers_version="4.6",
        pytorch_version="1.7",
        py_version="py36",
        hyperparameters={
            "epochs": 3,  # Number of training epochs
            "train-batch-size": 32,  # Batch size for training data
            "eval-batch-size": 64,  # Batch size for evaluation data
            "warmup_steps": 500,  # Number of warm-up steps
            "learning_rate": 5e-5,  # Learning rate
            "model_name": "roberta-base",  # Pre-trained model name
            "num_labels": 6,  # Number of output labels
        },
        distribution=distribution,  # Distributed training configuration
    )
except Exception as e:
    logging.error(f"Error setting up HuggingFace estimator: {str(e)}")
    raise e

# Define the S3 bucket and paths where your training and validation datasets are stored
BUCKET_NAME = "your_bucket_name"
S3_TRAIN_PATH = "path_to_your_train_data"
S3_VAL_PATH = "path_to_your_validation_data"

# Validate the S3 paths to ensure they exist
s3 = boto3.resource("s3")


def validate_s3_path(bucket, path):
    try:
        s3.Object(bucket, path).load()
    except Exception as e:
        logging.error(f"Error accessing S3 path s3://{bucket}/{path}: {str(e)}")
        raise e


validate_s3_path(BUCKET_NAME, S3_TRAIN_PATH)
validate_s3_path(BUCKET_NAME, S3_VAL_PATH)

# Define the S3 paths for the training and validation datasets
train_input = f"s3://{BUCKET_NAME}/{S3_TRAIN_PATH}"
validation_input = f"s3://{BUCKET_NAME}/{S3_VAL_PATH}"

# Start the SageMaker training job
try:
    huggingface_estimator.fit({"train": train_input, "validation": validation_input})
except Exception as e:
    logging.error(f"Error initiating SageMaker training job: {str(e)}")
    raise e
