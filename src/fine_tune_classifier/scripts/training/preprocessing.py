#!/usr/bin/env python3

import pandas as pd
import io
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
import boto3
import logging

from utils import read_s3_file

# Initialize logging
logging.basicConfig(level=logging.INFO)

BUCKET_NAME = "your_bucket_name"
OLID_DIR = "OLID"
s3_client = boto3.client('s3')

class Preprocessor:
    """
    Class to handle preprocessing of data for training.
    
    Attributes:
    - bucket_name (str): Name of the S3 bucket where data is stored.
    - olid_dir (str): Directory in the S3 bucket containing the OLID dataset.
    """

    def __init__(self, bucket_name, olid_dir):
        """
        Initializes the Preprocessor with the specified S3 bucket and directory.
        
        Args:
        - bucket_name (str): Name of the S3 bucket where data is stored.
        - olid_dir (str): Directory in the S3 bucket containing the OLID dataset.
        """
        self.bucket_name = bucket_name
        self.olid_dir = olid_dir
    
    def load_and_merge_data(self):
        """
        Loads the training data and labels from S3 and merges them based on ID.
        
        Returns:
        - DataFrame: Merged training data and labels.
        """
        # Load the training dataset from S3
        train_data = read_s3_file(self.bucket_name, f"{self.olid_dir}/olid-training-v1.0.tsv")

        # Load labels for 'subtask_a'
        labels_a = read_s3_file(self.bucket_name, f"{self.olid_dir}/labels-levela.csv")

        # Merge labels to training data based on ID
        return train_data.merge(labels_a, left_on="id", right_on="id")
    
    def tokenize_and_split_data(self, train_data):
        """
        Tokenizes the dataset and splits it into training and validation sets.
        
        Args:
        - train_data (DataFrame): Merged training data and labels.
        
        Returns:
        - Tuple: Tokenized training inputs, validation inputs, training labels, and validation labels.
        """
        # Tokenize the dataset using the Roberta tokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        tokenized_data = tokenizer(train_data['tweet'].tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt")

        # Split the dataset into training and validation sets
        return train_test_split(tokenized_data['input_ids'], train_data['subtask_a'].tolist(), test_size=0.2, random_state=42)

    def convert_to_datasets_format(self, train_inputs, val_inputs, tokenized_data, train_labels, val_labels):
        """
        Converts tokenized input and labels to the datasets format.
        
        Args:
        - train_inputs (Tensor): Tokenized training inputs.
        - val_inputs (Tensor): Tokenized validation inputs.
        - tokenized_data (Dict): Tokenized data.
        - train_labels (List): Training labels.
        - val_labels (List): Validation labels.
        
        Returns:
        - Tuple: Training dataset and validation dataset in datasets format.
        """
        # Convert tokenized data to datasets format
        train_dataset = Dataset.from_dict({
            'input_ids': train_inputs,
            'attention_mask': tokenized_data["attention_mask"][train_inputs.index],
            'labels': train_labels
        })

        val_dataset = Dataset.from_dict({
            'input_ids': val_inputs,
            'attention_mask': tokenized_data["attention_mask"][val_inputs.index],
            'labels': val_labels
        })

        return train_dataset, val_dataset
    
    def save_to_s3(self, train_dataset, val_dataset):
        """
        Saves the processed datasets back to S3.
        
        Args:
        - train_dataset (Dataset): Training dataset in datasets format.
        - val_dataset (Dataset): Validation dataset in datasets format.
        """
        S3_TRAIN_PATH = "train_data_path_on_s3/train_dataset"
        S3_VAL_PATH = "val_data_path_on_s3/val_dataset"

        # Save the datasets to S3
        train_dataset.save_to_disk(f"s3://{self.bucket_name}/{S3_TRAIN_PATH}")
        val_dataset.save_to_disk(f"s3://{self.bucket_name}/{S3_VAL_PATH}")

    def preprocess(self):
        """
        Main method to execute the preprocessing steps.
        """
        try:
            train_data = self.load_and_merge_data()
            train_inputs, val_inputs, train_labels, val_labels = self.tokenize_and_split_data(train_data)
            train_dataset, val_dataset = self.convert_to_datasets_format(train_inputs, val_inputs, train_labels, val_labels)
            self.save_to_s3(train_dataset, val_dataset)
            logging.info("Preprocessing completed and datasets saved to S3.")
        except Exception as e:
            logging.error(f"Error during preprocessing: {str(e)}")
            raise

if __name__ == "__main__":
    preprocessor = Preprocessor(BUCKET_NAME, OLID_DIR)
    preprocessor.preprocess()

