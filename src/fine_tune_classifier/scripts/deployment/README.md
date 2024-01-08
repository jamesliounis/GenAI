# Overview
The deployment directory contains scripts essential for deploying the machine learning model on Amazon SageMaker.

# Contents

`deploy_model.py`:

- This script provides a mechanism to deploy a pre-trained model on Amazon SageMaker.
- Initializes a SageMaker session and sets up the PyTorch model configuration for deployment.
- The model artifact location in S3 and the SageMaker IAM role need to be specified before running the script.
After setting up, it deploys the model to a SageMaker endpoint.

`inference.py`:

- Essential for SageMaker inference.
- Contains the model_fn function to load the model and tokenizer during SageMaker initialization. Includes a model warm-up step to ensure low-latency for the first prediction.
- The `input_fn` function preprocesses the incoming data. It currently supports content types text/plain and application/json.
- The `predict_fn` function makes predictions using the preprocessed data.
- Enhanced with error handling to gracefully handle potential issues.

# Usage

## Deployment:

1. Update the `s3_model_path` in `deploy_model.py` to point to your model artifact in S3.
2. Update the `role` parameter in `deploy_model.py`` with your SageMaker IAM role ARN.
3. Run the `deploy_model.py` script to deploy the model to a SageMaker endpoint:

```shell script
./deploy_model.py
```
## Inference:

- Once deployed, the model will use the functions in inference.py to handle incoming requests.
- Requests can be of content type text/plain (plain text string) or application/json (a list or a dictionary with a 'text' key).






