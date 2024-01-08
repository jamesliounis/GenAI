import boto3
import pandas as pd
import io

def read_s3_file(bucket_name, file_path):
    """Read a file from S3 and return as a pandas DataFrame."""
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_path)
    return pd.read_csv(io.BytesIO(obj['Body'].read()), delimiter="\t" if file_path.endswith(".tsv") else ",")
