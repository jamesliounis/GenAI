#!/usr/bin/env python3

import sagemaker
from sagemaker.pytorch import PyTorchModel

# Initialize a SageMaker session
sagemaker_session = sagemaker.Session()

# Specify the S3 path to the model artifact ('model.tar.gz').
# This should be replaced with the actual S3 path where the model artifact resides.
s3_model_path = "s3://path/to/model.tar.gz"

# Create a PyTorch model for deployment.
# - model_data: Points to the S3 path of the model artifact.
# - role: The SageMaker IAM role ARN. This should be replaced with the actual ARN.
# - framework_version: The version of PyTorch being used.
# - py_version: The version of Python being used.
# - entry_point: The script that will be executed for inference when the model is deployed.
model = PyTorchModel(
    model_data=s3_model_path,
    role="your_sagemaker_role_arn",
    framework_version="1.8.0",  # or your specific PyTorch version
    py_version="py3",
    entry_point="inference.py",
)

# Deploy the model to a SageMaker endpoint.
# - instance_type: Specifies the type of EC2 instance to deploy the model.
# - initial_instance_count: The number of instances to start with.
predictor = model.deploy(instance_type="ml.m5.large", initial_instance_count=1)
