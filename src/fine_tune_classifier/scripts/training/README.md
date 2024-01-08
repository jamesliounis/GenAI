# Launching SageMaker Training Jobs

## Overview

This directory contains scripts that handle the training phase of the text classification pipeline. The primary components are:

`launch_sagemaker_training.py`: This script is equipped to launch SageMaker training jobs on either single or distributed GPU instances. By leveraging SageMaker's capabilities, it offers flexible infrastructure options for training, such as data parallelism for distributed training. The script internally configures the HuggingFace estimator based on the mode selected ("distributed" or "single-instance").

`preprocessing.py`: This script handles the preprocessing phase of the pipeline. It loads the raw data, tokenizes it using the Roberta tokenizer, splits it into training and validation sets, and then saves these datasets in a format ready for training.

`train.py`: This is the core training script that is called by both the distributed and single-instance training launchers. It sets up the model, training arguments, and trainer. It also handles the training and evaluation phases, saves the evaluation results, and finally packages and uploads the trained model to S3.

## Usage

1. Setting Up:

Ensure your AWS credentials are set up correctly.
Replace placeholders in the script such as 'your_bucket_name', 'path_to_your_train_data', and 'path_to_your_validation_data' with appropriate values.

2. Preprocessing

```shell script
python preprocessing.py
```

3. Launching a SageMaker Training Job:

For single-instance training:
```shell script
python launch_training.py --mode single-instance
```

For distributed (multi-instance) training:
```shell script
python launch_training.py --mode distributed
```

Modify hyperparameters and training configurations as needed.

Ensure you select a multi-GPU instance type and specify the number of instances for distributed training. You may want to consider various [instance types](https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-available-instance-types.html) and their [pricing](https://aws.amazon.com/sagemaker/pricing/).

**Notes**:

- Always ensure data is preprocessed using the preprocessing.py script before initiating the SageMaker training job.
- The SageMaker training script integrates seamlessly with the AWS ecosystem, including S3 for data storage and retrieval.
- If there are any issues during initialization or training, they will be logged for troubleshooting.
- The scripts are designed to work seamlessly with AWS SageMaker, but can be modified for other platforms or local training as needed.

