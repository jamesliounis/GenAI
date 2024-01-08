#!/usr/bin/env python3

import torch
from model import CombinedModel
from tokenize_reviews import tokenize_reviews

# Sample product reviews
reviews = [
    "I love this product! Best purchase this year.",
    "It's okay. Not the best, but not the worst either.",
    "Really disappointed. The product broke after one use.",
    "This is amazing! Highly recommend.",
    "Meh, it was okay. Not what I expected.",
    "The best product in its category!",
    "I wish I hadn't bought this.",
    "Decent product for its price."
]

# Tokenize the reviews
input_ids, attention_mask = tokenize_reviews(reviews)

# For numerical data
batch_size = len(reviews)
num_numerical_features = 5  # For demonstration purposes; adjust based on your actual data
numerical_data = torch.randn(batch_size, num_numerical_features)

# Example numerical data might represent:
# - Number of previous products purchased by the reviewer
# - Time spent on the review page
# - Product price
# - Number of times the review was viewed
# - Average rating of other reviews by the same user

# For categorical data
num_categories = [10, 5, 3]  # Three categorical features: 10, 5, and 3 unique categories respectively
categorical_data = torch.stack([
    torch.randint(0, num_categories[i], (batch_size,)) for i in range(len(num_categories))
], dim=1)

# Example categorical data might represent:
# - Product category (e.g., electronics, clothing, books)
# - User membership status (e.g., regular, premium, VIP)
# - Source of the review (e.g., mobile, desktop, app)

# Initialize the model
model = CombinedModel(num_numerical_features, num_categories)

# Pass the sample inputs through the model
outputs = model(input_ids, attention_mask, numerical_data, categorical_data)

# Print the outputs
print(outputs)

# Each value should be between 0 and 1, due to the sigmoid activation in the output layer, 
# which indicates the model's prediction for each review. 
# For a binary classification task, values closer to 1 indicate a positive review (or "good"), 
# and values closer to 0 indicate a negative review (or "bad").
