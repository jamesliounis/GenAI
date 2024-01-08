#!/usr/bin/env python3

import torch
from transformers import DistilBertTokenizer
from model import CombinedModel
from tokenize_reviews import tokenize_reviews  # Import our custom tokenizer function

def load_model(model_path):
    """
    Load the trained model from a specified path.

    Parameters:
    -----------
    model_path : str
        Path to the model's saved weights.

    Returns:
    --------
    model : CombinedModel
        The trained model in evaluation mode.
    """
    # Initialize the model with specified features
    model = CombinedModel(num_numerical_features=5, num_categories=[10, 5, 3])
    
    # Load the model's saved weights
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode (this affects layers like dropout)
    model.eval()
    
    return model

def predict(model, text_data, numerical_data, categorical_data):
    """
    Make predictions on new data using the trained model.

    Parameters:
    -----------
    model : CombinedModel
        The trained model.
    text_data : list of str
        List of product reviews.
    numerical_data : torch.Tensor
        Tensor containing numerical features.
    categorical_data : torch.Tensor
        Tensor containing categorical data.

    Returns:
    --------
    outputs : torch.Tensor
        Model's predictions for the input data.
    """
    
    # Convert raw text data to model-friendly format
    inputs = tokenize_reviews(text_data)
    
    # Ensure no gradients are computed for this forward pass (improves efficiency)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'], numerical_data, categorical_data)
    return outputs

if __name__ == "__main__":
    # Define path to the saved model
    model_path = "path/to/s3_bucket/trained_models/model.pth"
    
    # Load the trained model
    model = load_model(model_path)

    # TODO: Fetch your test data from the cloud or other storage
    # For demonstration, below are placeholders for paths to Cloud storage
    text_data_path = "path/to/cloud/text_data"
    numerical_data_path = "path/to/cloud/numerical_data"
    categorical_data_path = "path/to/cloud/categorical_data"
    
    # Load data from the Cloud paths (you'd replace this with actual Cloud fetch logic)
    text_data = ...  # Load from text_data_path
    numerical_data = ...  # Load from numerical_data_path
    categorical_data = ...  # Load from categorical_data_path

    # Get predictions for the test data
    predictions = predict(model, text_data, numerical_data, categorical_data)
    
    # Print the predictions
    print(predictions)
