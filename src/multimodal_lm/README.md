# Multimodal Language Model for Product Review Analysis

Welcome to the repository for the Multimodal Language Model, designed to process and analyze product reviews! The model synergizes textual data with numerical and categorical data for comprehensive predictions.

## Table of Contents

- [Multimodal Language Model for Product Review Analysis](#multimodal-language-model-for-product-review-analysis)
  - [Why Multimodal?](#why-multimodal)
  - [Repository Structure](#repository-structure)
  - [Getting Started](#getting-started)
  - [Diving into the Solution](#diving-into-the-solution)
    - [model.py](#modelpy)
    - [Dockerfile](#dockerfile)
    - [example_usage.py](#example_usagepy)
    - [predict.py](#predictpy)
    - [requirements.txt](#requirementstxt)
    - [train.py](#trainpy)
  - [Generating Sample Predictions (optional)](#generating-sample-predictions-optional)
  - [Step-by-step guidance for Your Use Case](#step-by-step-guidance-for-your-use-case)
    - [Gather your data](#gather-your-data)
    - [Tokenize your reviews](#tokenize-your-reviews)
    - [Train the Model](#train-the-model)
    - [Model Testing / Generating Predictions](#model-testing--generating-predictions)
    - [Additional Guidance: Training and Deploying the Model on AWS](#additional-guidance-training-and-deploying-the-model-on-aws)
      - [Training the Model on SageMaker](#training-the-model-on-sagemaker)
      - [Deploying the Model](#deploying-the-model)


## Why Multimodal?

While textual reviews offer insights into customer sentiments, numerical and categorical data such as purchase history and time spent on a review can further enrich our understanding. By combining all these data types, we aim to provide a more accurate gauge of the review's sentiment, outperforming models that only analyze one type of data.

## Repository Structure
-----------------------

```
.
└── src
    └── multimodal_lm
        ├── Dockerfile
        ├── example_usage.py
        ├── model.py
        ├── predict.py
        ├── requirements.txt
        ├── tokenize_reviews.py
        └── train.py
```
----------------------

## Getting Started

1. **Clone the Repository**

You may wish to do this in a virtual environment as follows:
```shell script
# Install virtualenv if you haven't already
pip install virtualenv

# Create a new virtual environment named 'venv' or any name you prefer
virtualenv venv

# Activate the virtual environment
# For Windows: .\venv\Scripts\activate
# For MacOS/Linux:
source venv/bin/activate

# Now, clone the repository and navigate to the desired directory
git clone git@github.com:jamesliounis/HuggingFace-TakeHome.git
cd src/multimodal_lm
```

## Diving into the Solution

1. `model.py`

Defines our **Multimodal Language Model**. We've chosen [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) for textual processing due to its efficiency without compromising too much on accuracy. This model seamlessly integrates textual data with other types for a complete analysis.

2. `Dockerfile`

Provides an environment-agnostic way to execute our solution. Ensures that irrespective of where you run, the results and behavior remain consistent.

3. `example_usage.py`

A hands-on script showcasing how to utilize the model. Sample product reviews alongside numerical and categorical data are processed to provide a sentiment score.

4. `predict.py`

A utility to load a pre-trained model and predict sentiments on new reviews and associated data.

5. `requirements.txt`

All the Python package dependencies are listed here.

6. `train.py`

This script fetches data, trains the model, and saves it for future use. It saves the model at the end of every epoch. 

7. `preprocess.py`

Contains data preprocessing functions required to transform and clean raw data into a format suitable for training and inference. The TODO comments in the script indicate areas where custom data preprocessing steps should be implemented to prepare the data for the machine learning model.

## Generating Sample Predictions (optional)
   
1. Ensure Docker and Docker Compose are installed:
   - [Docker](https://docs.docker.com/get-docker/)

2. **Pull the Docker Image**

```shell script
docker pull jamesliounis/hugging_face_takehome:multimodal_lm
```

3. **Run the Container to See Sample Predictions**

```shell script
docker run jamesliounis/hugging_face_takehome:multimodal_lm
```


## Step-by-step guidance for Your Use Case

1. **Gather your data**

- Prepare your textual reviews. 
- Organize numerical data (e.g., number of products a user has purchased or time spent looking at a review).
- Collate categorical data (e.g., representing user membership tiers or sources of the review). 

Note: our scripts assume that you have your data stored in an Amazon S3 bucket, which is generally good practice. However, feel free to use the method that works best for your use case. 

2. **Tokenize your reviews**

Before training or prediction, you'll need to convert your textual reviews into a format suitable for the model. Use the `tokenize_reviews.py` script for this purpose.

3. **Train the Model**:

- With your data prepared, the next step is to train the model.
- If your data is hosted on an S3 bucket, adjust the constants in `train.py` to reflect the path to your data and the desired model save path.
```shell script
S3_BUCKET = "your_bucket_name"
DATA_PATH = "path_to_your_training_data"
SAVE_PATH = "trained_models/your_model_name"
```
- Add any extra data pre-processing steps that you may require. 
- Execute the `train.py` script to start the training process:
```shell script
python train.py
```

4. **Model Testing / Generating Predictions**:

After training, you'd want to see how the model performs on new reviews.
- Use the `predict.py` script to load the trained model and get predictions on new reviews.
- Make sure to adjust the paths in the script to reflect the location of your trained model.
```shell script
model_path = "path/to/s3_bucket/trained_models/model.pth"
```
Execute the script to see predictions:
```shell script
python predict.py
```
The predictions, between 0 and 1, will give you an understanding of whether the review leans positive or negative.

5. **Additional Guidance: Training and Deploying the Model on AWS**

Training the Model on SageMaker:
[SageMaker](https://www.googleadservices.com/pagead/aclk?sa=L&ai=ClHen3LApZcmqOcqXnboPv8Gg2AX11cbHc476y8muEcaUtu7XCwgAEAEguVRgyZbuiISk7A-gAZ_tnsYDyAEByAPYIKoEV0_QN9n7TOPXko_1wKEd0h9YpcYojTG6scARSqG_xVpRxFmY8WPpx8RU3clZww28Nci3nfjXA5piyFqqwz3_Rkzru-KBa3ZLji_g_EcChKWPexOIUwM_DcAE-Y7HnpwEgAWQTogF1qu--kmgBmaAB4rc-QaIBwGQBwGoB6a-G6gHvK2xAqgHuauxAqgHuZqxAqgH89EbqAfu0huoB_-csQKoB8rcG6gHkq-xAqgHu6SxAqgHkqaxAqgH2KaxAqgH3rWxAqgH26qxAqgH3LCxAqgHv7mxAqgH6rGxAqgHqrixAqgHlLixAqgH7LixAqgHvrexAqAIj7ilBLAIAdIIJhACMgSD4IAOOgnC4YCAgIAEgkBCAQRI0ezzKlAJWMmI3bf284EDmgkkaHR0cHM6Ly9hd3MuYW1hem9uLmNvbS9wbS9zYWdlbWFrZXIvsQkELeDXrED3urkJ8gLGq76k6Qf4CQGKCoUCaHR0cHM6Ly9waXhlbC5ldmVyZXN0dGVjaC5uZXQvNDQyMi9jcT9ldl9zaWQ9MyZldl9sbj1hbWF6b24lMjBzYWdlbWFrZXImZXZfbHg9a3dkLTQwMTU0MjU4MDgwNiZldl9jcng9NjUxNzUxMDYwNjkyJmV2X210PWUmZXZfbj1nJmV2X2x0eD0mZXZfcGw9JmV2X3Bvcz0mZXZfZHZjPWMmZXZfZHZtPSZldl9waHk9OTAwMTk5OSZldl9sb2M9JmV2X2N4PTE5ODUyNjYyMjMwJmV2X2F4PTE0NTAxOTIyNTk3NyZldl9lZmlkPXtnY2xpZH06RzpzJnVybD17bHB1cmx9mAsB4AsBqgwCCAHoDAaqDQJVU8gNAYIUFAgDEhBhbWF6b24gc2FnZW1ha2VyyBSUweCz-bKkh3DQFQGYFgH4FgGAFwGSFwgSBggBEAMYA-AXAg&ae=2&gclid=Cj0KCQjw1aOpBhCOARIsACXYv-dclwiWvIuLFyhBw_ORT2bxbsL7H_zUZZtxQwbgh_J1_EeksbcuHtkaAug7EALw_wcB&ved=2ahUKEwjR-dW39vOBAxXurYkEHc8jAvEQ0Qx6BAgGEAE&nis=8&dct=1&cid=CAASFeRoh5-bDE03f5E7FDz8LXYUdPHixA&dblrd=1&sival=AF15MEAbf10EPX6uKihMD_LNRAr-Z0Xd6LRTqmSJ1D_DOZglTpQxk2rgr1A_VStyLxPFDfzU8Kou0zUoRq_EAZkQ8SxBrHFAJco71Dx4LUUQhtbB0lJsv_fKTUzAmSUf4isHak6yaFyTOoa1oz4ZqGn_-0QdIbA80DlGPjqs0i4kOWOPPUpMJFU&sig=AOD64_0dgtuOSqQHDBLf3uaQNjULjnd_XA&adurl=https://aws.amazon.com/pm/sagemaker/%3Ftrk%3Db6c2fafb-22b1-4a97-a2f7-7e4ab2c7aa28%26sc_channel%3Dps%26ef_id%3D%7Bgclid%7D:G:s%26s_kwcid%3DAL!4422!3!651751060692!e!!g!!amazon%2520sagemaker!19852662230!145019225977) is a fully managed service that provides developers and data scientists the ability to build, train, and deploy machine learning (ML) models quickly.

- Data Preparation: Store your training and validation data in an S3 bucket.
- Set Up the Environment:
Open the SageMaker console on AWS.
Create a new Jupyter notebook instance.
- Training Script: Modify the training script to use SageMaker's environment variables for accessing data and saving models. You would need to use `os.environ['SM_CHANNEL_TRAIN']` for training data location and `os.environ['SM_MODEL_DIR']` to save your models.
- Start Training:
Define SageMaker Estimators with the necessary configurations (instance type, role, entry point to your script, framework version) and call the fit method of the estimator, passing in the S3 location of your data.
- Monitor Training: Use SageMaker's built-in monitoring tools to visualize training metrics and logs.

Deploying the Model:
   
[AWS Lambda](https://aws.amazon.com/lambda/) and [API Gateway](https://aws.amazon.com/api-gateway/) are recommended for deploying the model as they offer a serverless approach, simplifying deployment and scaling.

- Model Export: Ensure your model is exported in a format compatible with your inference code (e.g., TorchScript for PyTorch models).
- Create a Lambda Function:
Navigate to AWS Lambda in the AWS Console.
Create a new function and upload your model along with the necessary inference code.
Increase the function's timeout and memory depending on the model's size and inference time.
- API Gateway:
Open the API Gateway console on AWS.
Create a new API and link it to your Lambda function.
Configure the necessary routes (e.g., a POST route for model predictions).
Test Deployment: Use tools like Postman or simple curl commands to test the API endpoint.
Secure & Monitor: Ensure you've set up necessary security measures (like API keys or IAM roles) and monitor API usage and response times.
Note: these steps provide a high-level overview. However, each of them may have additional configurations or best practices specific to the use-case or data at hand.

## Conclusion

This solution is tailored for businesses looking to harness both textual and non-textual data for richer insights from their reviews. Combining these data sources promises a more nuanced understanding, potentially outshining models that only consider one data type.




