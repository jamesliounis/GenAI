# Overview

The inference directory contains the script essential for making predictions using a model deployed on Amazon SageMaker.

# Contents

`infer.py`:
- A utility script to make predictions using a deployed SageMaker endpoint.
- Uses the boto3 client for SageMaker Runtime to invoke a specified endpoint.
- Can handle different content types for the payload, with application/json set as the default.
- Enhanced with error handling and logging to gracefully manage potential issues and provide detailed feedback.
- Command-line interface provided for users to easily input text data for predictions and optionally specify the SageMaker endpoint name.

# Usage

## Making Predictions:

1. Ensure the AWS CLI is set up with appropriate credentials or your environment is configured with the necessary AWS credentials.
2. Modify `DEFAULT_ENDPOINT_NAME` in `infer.py` to the name of your SageMaker endpoint if you don't want to provide it via command line every time.
3. Run the script using the command:
```shell script
python infer.py --input-text "Your text here for prediction."
```
4. (Optional) Specify the SageMaker endpoint name with the `--endpoint-name` argument if different from the default.