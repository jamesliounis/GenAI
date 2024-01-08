#!/usr/bin/env python3

import boto3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import CombinedModel
from tokenize_reviews import tokenize_reviews
from sklearn.preprocessing import StandardScaler
from preprocess import custom_data_preprocessing
from torch.optim import AdamW


# Constants
# Specify the S3 bucket name, data path, and save path for the trained model.
S3_BUCKET = "your_bucket_name"
DATA_PATH = "path_to_your_training_data"
SAVE_PATH = "trained_models/your_model_name"
BATCH_SIZE = 32  # Number of training samples processed before the model's internal parameters are updated.
EPOCHS = 5  # Number of complete passes over the entire dataset.
LEARNING_RATE = 5e-5  # Determines the step size at each iteration while moving towards a minimum of the loss function.

# Set up the computational device for PyTorch. Use GPU if available, else use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the StandardScaler for scaling numerical data to have mean=0 and variance=1.
scaler = StandardScaler()


class ReviewDataset(Dataset):
    """
    Custom dataset to load and preprocess review data from S3.
    """

    def __init__(self, s3_path):
        """
        Fetches and processes data from S3 upon initialization.

        Parameters:
        -----------
        s3_path : str
            Path to the data file in the S3 bucket.
        """
        # Initialize the S3 client and fetch raw data.
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_path)
        raw_data = obj["Body"].read().decode("utf-8")

        # Process the raw data using the custom data preprocessing function.
        # TODO: customize pre-processing function in `preprocess.py`
        self.data = custom_data_preprocessing(raw_data)

        # Scale the numerical data to have zero mean and unit variance.
        self.data["numerical_data"] = scaler.fit_transform(self.data["numerical_data"])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        --------
        int
            Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset given an index.

        Parameters:
        -----------
        idx : int
            Index of the desired sample.

        Returns:
        --------
        dict
            Processed sample data.
        """
        # Get the raw data for the sample associated with the index.
        raw_data_for_sample = self.data[idx]

        # Use the custom data preprocessing function to preprocess the raw data.
        processed_data = custom_data_preprocessing(raw_data_for_sample)

        # Tokenize the review text using the custom tokenizer.
        review_text = processed_data["review"]
        input_ids, attention_mask = tokenize_reviews(review_text)

        # Additional data processing if needed...
        # You can process other fields of processed_data here.

        # Return the processed sample as a dictionary.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "numerical_data": processed_data["numerical_data"],
            "categorical_data": processed_data["categorical_data"],
            "labels": processed_data["labels"]  # Replace with the actual label field
        }



# Load the dataset and prepare a DataLoader for batching and shuffling.
dataset = ReviewDataset(DATA_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model, optimizer, and loss function.
model = CombinedModel(num_numerical_features=2, num_categories=[2, 2]).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss().to(device)

# Training loop.
for epoch in range(EPOCHS):
    model.train()  # Set the model to training mode.
    total_loss = 0
    for batch in dataloader:
        # Load data from the current batch to the computational device.
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        numerical_data = scaler.transform(batch["numerical_data"].numpy()).to(device)
        categorical_data = batch["categorical_data"].to(device)
        labels = batch["labels"].to(device)

        # Zero out the gradients.
        optimizer.zero_grad()

        # Forward pass: compute the model's predictions.
        outputs = model(input_ids, attention_mask, numerical_data, categorical_data)

        # Compute the loss.
        loss = loss_fn(outputs, labels)

        # Backward pass: compute the gradient of the loss with respect to model parameters.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        total_loss += loss.item()

    # Print the average loss for the current epoch.
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader)}")

    # Save the trained model after each epoch.
    model_save_path = f"model_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_save_path)
    try:
        s3_client.upload_file(model_save_path, S3_BUCKET, SAVE_PATH + "/" + model_save_path)
    except Exception as e:
        print(f"Error uploading model: {e}")
    else:
        print(f"Model {model_save_path} uploaded successfully.")

    # Save the final trained model with a descriptive filename.
    final_model_save_path = "final_trained_model.pth"
    torch.save(model.state_dict(), final_model_save_path)
    try:
        s3_client.upload_file(final_model_save_path, S3_BUCKET, SAVE_PATH + "/" + final_model_save_path)
    except Exception as e:
        print(f"Error uploading final model: {e}")
    else:
        print(f"Final model {final_model_save_path} uploaded successfully.")


