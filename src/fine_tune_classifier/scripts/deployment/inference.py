#!/usr/bin/env python3

import os
import json
from transformers import RobertaForSequenceClassification, RobertaTokenizer


def model_fn(model_dir):
    """
    Load the Roberta model and tokenizer.

    Args:
    - model_dir: Path to the directory where the model is saved.

    Returns:
    - Tuple of (model, tokenizer)
    """
    try:
        model = RobertaForSequenceClassification.from_pretrained(model_dir)
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Model warm-up
        sample_input = "This is a sample input for model warm-up."
        inputs = tokenizer(
            sample_input,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        _ = model(**inputs)

        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")


def input_fn(request_body, request_content_type):
    """
    Preprocess the input data for prediction.

    Args:
    - request_body: The payload of the inference request.
    - request_content_type: Content type of the request.

    Returns:
    - Preprocessed input data.
    """
    try:
        # Handle different content types
        if request_content_type == "application/json":
            input_data = json.loads(request_body)
            if isinstance(input_data, list):
                return input_data[0]  # Assume the first element is the input text
            elif isinstance(input_data, dict) and "text" in input_data:
                return input_data["text"]
            else:
                raise ValueError(
                    "Unexpected JSON format. Expected a list or a dictionary with a 'text' key."
                )
        elif request_content_type == "text/plain":
            return request_body
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        raise RuntimeError(f"Error preprocessing input data: {str(e)}")


def predict_fn(input_data, model):
    """
    Make predictions on the input data using the model.

    Args:
    - input_data: Preprocessed input data.
    - model: Tuple of (model, tokenizer)

    Returns:
    - Predicted class for the input data.
    """
    try:
        model, tokenizer = model
        inputs = tokenizer(
            input_data,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        outputs = model(**inputs)
        preds = outputs.logits.argmax(-1).item()
        return preds
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")
