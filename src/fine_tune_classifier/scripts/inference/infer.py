#!/usr/bin/env python3

import argparse
import boto3
import logging

# Initialize boto3 client for SageMaker Runtime
sagemaker_runtime = boto3.client("sagemaker-runtime")

# Default configurations for the SageMaker endpoint
DEFAULT_ENDPOINT_NAME = "your-sagemaker-endpoint-name"
CONTENT_TYPE = "application/json"  # Can be adjusted based on the type of request payload expected by the endpoint

# Initialize logging
logging.basicConfig(level=logging.INFO)


def make_prediction(
    endpoint_name: str, input_payload: str, content_type: str = CONTENT_TYPE
):
    """
    Make a prediction using a deployed SageMaker model.

    Args:
    - endpoint_name: Name of the deployed SageMaker endpoint.
    - input_payload: Data to be sent to the model for prediction.
    - content_type: Type of the data being sent to the endpoint. Default is 'application/json'.

    Returns:
    - Predicted result from the model.
    """
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name, ContentType=content_type, Body=input_payload
        )

        result = response["Body"].read().decode("utf-8")
        return result

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return None


# Main execution block to parse command-line arguments and make predictions
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script to make predictions using a deployed model endpoint."
    )
    parser.add_argument(
        "--input-text",
        type=str,
        required=True,
        help="Input text for which to make a prediction.",
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=DEFAULT_ENDPOINT_NAME,
        help="Name of the SageMaker deployed endpoint.",
    )

    args = parser.parse_args()

    # Format input data as payload for endpoint. Adjust this according to the payload format expected by your endpoint.
    payload = {"text": args.input_text}

    # Make a prediction using the deployed model endpoint
    predicted_result = make_prediction(args.endpoint_name, str(payload))

    if predicted_result:
        print(f"Predicted Result: {predicted_result}")
    else:
        print("Failed to obtain prediction.")
