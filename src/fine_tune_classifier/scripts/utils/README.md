# Overview

The utils directory contains utility functions that provide essential support to other modules within the project. These utility functions simplify common operations and maintain cleaner code in the main scripts by abstracting out repetitive tasks.

## Files & Descriptions
`utils.py`:
Contains utility functions for various operations.

`read_s3_file(bucket_name, file_path)`: This function reads a file from Amazon S3 and returns its content as a pandas DataFrame. The delimiter for reading the CSV is determined based on the file's extension.

## Usage

To use functions from the utils module:

```shell script
from scripts.utils import read_s3_file
```