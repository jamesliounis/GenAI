# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the local contents into the container
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "example_usage.py"]

