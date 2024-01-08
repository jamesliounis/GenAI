#!/usr/bin/env python3

import torch
import torch.nn as nn
from transformers import DistilBertModel

class CombinedModel(nn.Module):
    """
    A multimodal deep learning model that merges textual data, processed by DistilBERT, 
    with numerical and categorical data for binary classification tasks.
    """
    
    def __init__(self, num_numerical_features, num_categories, 
             text_out_features=16, num_layers=1, num_neurons_per_layer=[16], 
             cat_layers=1, cat_neurons_per_layer=[16], combined_out_features=16, 
             cat_embedding_size=8, dropout_prob=0.1, num_labels=1, cat_embedding_sizes=None):
        """
        Initialize the model components.
        
        Parameters:
        -----------
        num_numerical_features : int
            Number of numerical features in the input.
        num_categories : list of int
            A list containing the number of unique categories for each categorical feature.
        text_out_features : int, optional
            Output feature size for textual data linear layer. Default is 16.
        num_out_features : int, optional
            Output feature size for numerical data linear layer. Default is 16.
        cat_out_features : int, optional
            Output feature size for categorical data linear layer. Default is 16.
        combined_out_features : int, optional
            Output feature size for combined data linear layer. Default is 16.
        cat_embedding_size : int, optional
            Embedding size for categorical data if no specific sizes are provided. Default is 8.
        dropout_prob : float, optional
            Dropout probability for regularization. Default is 0.1.
        num_labels : int, optional
            Number of output labels, default is 1 for binary classification.
        cat_embedding_sizes : list of int, optional
            List of embedding sizes for each categorical feature. 
            If not provided, cat_embedding_size is used for all.
        num_layers : int, optional
            Number of layers for processing numerical data. Default is 1.
        num_neurons_per_layer : list of int, optional
            List of neurons for each layer in numerical data processing. Default is [16].
        cat_layers : int, optional
            Number of layers for processing categorical data. Default is 1.
        cat_neurons_per_layer : list of int, optional
            List of neurons for each layer in categorical data processing. Default is [16].
        """
        # Inherit the properties and methods from parent class, nn.Module
        super(CombinedModel, self).__init__()

        # Ensure number of neurons specified for each layer matches number of layers.
        # This is to prevent any mismatched dimensions when constructing the model.
        assert len(num_neurons_per_layer) == num_layers, "Mismatch between num_layers and length of num_neurons_per_layer"
        assert len(cat_neurons_per_layer) == cat_layers, "Mismatch between cat_layers and length of cat_neurons_per_layer"

        # Initialize DistilBERT to process textual data (lighter version of BERT
        # chosen for performance benefits without significant accuracy loss).
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Prevent parameters from being trained. Ensures that pre-trained knowledge
        # is retained and not modified during the training of our combined model.
        # Set to 'True' for fine-tuning. 
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # Linear layer to condense the output from DistilBERT into a fixed size (text_out_features).
        self.text_fc = self._create_linear_layer(self.distilbert.config.dim, text_out_features, dropout_prob)

        # Initializing processing layers for numerical data. These layers transform the raw numerical features
        # into a more abstract representation.
        num_previous = num_numerical_features
        self.numerical_layers = nn.ModuleList()
        for neurons in num_neurons_per_layer:
            self.numerical_layers.append(self._create_linear_layer(num_previous, neurons, dropout_prob))
            num_previous = neurons

        # Initialize embedding layers for categorical data. Embeddings transform categorical data into dense vectors.
        # If specific embedding sizes for each category are not provided, use default size (cat_embedding_size).
        if cat_embedding_sizes is None:
            cat_embedding_sizes = [cat_embedding_size] * len(num_categories)
        assert len(cat_embedding_sizes) == len(num_categories), "Embedding sizes list must match number of categorical features."

        # Store each categorical embedding in a ModuleList for easy iteration / processing.
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories[i], cat_embedding_sizes[i]) 
            for i in range(len(num_categories))
        ])

        # Initialize processing layers for embedded categorical data.
        cat_previous = sum(cat_embedding_sizes)
        self.cat_layers = nn.ModuleList()
        for neurons in cat_neurons_per_layer:
            self.cat_layers.append(self._create_linear_layer(cat_previous, neurons, dropout_prob))
            cat_previous = neurons

        # Combine the outputs from text, numerical, and categorical data into unified representation.
        combined_input_features = text_out_features + num_neurons_per_layer[-1] + cat_neurons_per_layer[-1]
        self.combined_fc = self._create_linear_layer(combined_input_features, combined_out_features, dropout_prob)

        # Final output layer for binary classification. 
        # Sigmoid activation function ensures output between 0 and 1.
        self.output = nn.Sequential(
            nn.Linear(combined_out_features, num_labels),
            nn.Sigmoid()
        )


    def _create_linear_layer(self, input_features, output_features, dropout_prob):
        """
        Utility function to create a linear layer followed by a dropout and a ReLU activation.
        
        Parameters:
        -----------
        input_features : int
            Number of input features for the linear layer.
        output_features : int
            Number of output features for the linear layer.
        dropout_prob : float
            Dropout probability.
            
        Returns:
        --------
        torch.nn.Sequential
            A sequential layer consisting of a Linear layer, followed by Dropout and ReLU activation.
        """
        return nn.Sequential(
            nn.Linear(input_features, output_features),  # Linear transformation of the data.
            nn.BatchNorm1d(output_features),  # Normalize the output of the linear layer.
            nn.Dropout(dropout_prob),  # Regularize by randomly setting some outputs to 0.
            nn.ReLU()  # Activation function to introduce non-linearity.
        )

    def handle_textual_data(self, input_ids, attention_mask):
        """
        Process textual data using DistilBERT and a subsequent linear layer.
        
        Parameters:
        -----------
        input_ids : torch.Tensor
            Tensor of token ids to be fed to DistilBERT.
        attention_mask : torch.Tensor
            Tensor representing attention masks for input tokens.
            
        Returns:
        --------
        torch.Tensor
            Processed textual data.
        """
        # Get DistilBERT's output for the given text.
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # Process the '[CLS]' token's vector with the linear layer.
        return self.text_fc(outputs.last_hidden_state[:, 0])

    def handle_numerical_data(self, numerical_data):
        """
        Process numerical data using a linear layer.
        
        Parameters:
        -----------
        numerical_data : torch.Tensor
            Tensor of numerical data.
            
        Returns:
        --------
        torch.Tensor
            Processed numerical data.
        """
        # Iterate through each defined linear layer and process numerical data
        for layer in self.numerical_layers:
            numerical_data = layer(numerical_data)
        return numerical_data

    def handle_categorical_data(self, categorical_data):
        """
        Process categorical data using embeddings and a subsequent linear layer.
        
        Parameters:
        -----------
        categorical_data : torch.Tensor
            Tensor of categorical data.
            
        Returns:
        --------
        torch.Tensor
            Processed categorical data.
        """
        # Convert raw categorical values into embeddings
        # Retrieve corresponding embedding 
        # from embedding layer for each feature in categorical data
        embeddings = [embedding(categorical_data[:, i]) for i, embedding in enumerate(self.embeddings)]
        
        # Concatenate embeddings along feature dimension
        embeddings = torch.cat(embeddings, 1)

        # Process concatenated embeddings through each defined linear layer
        for layer in self.cat_layers:
            embeddings = layer(embeddings)

        return embeddings

    def forward(self, input_ids, attention_mask, numerical_data, categorical_data):
        """
        Forward pass for the combined model.
        
        Parameters:
        -----------
        input_ids : torch.Tensor
            Tensor of token ids to be fed to DistilBERT.
        attention_mask : torch.Tensor
            Tensor representing attention masks for input tokens.
        numerical_data : torch.Tensor
            Tensor of numerical data.
        categorical_data : torch.Tensor
            Tensor of categorical data.
            
        Returns:
        --------
        torch.Tensor
            Final output after processing all data types.
        """
        # Process textual data using DistilBERT and subsequent linear layer
        text_outputs = self.handle_textual_data(input_ids, attention_mask)

        # Process numerical data through its designated layers
        numerical_outputs = self.handle_numerical_data(numerical_data)

        # Convert categorical data to embeddings and process through its designated layers
        cat_outputs = self.handle_categorical_data(categorical_data)

        # Concatenate outputs from all data types to form unified representation
        combined = torch.cat((text_outputs, numerical_outputs, cat_outputs), 1)

        # Process combined representation through final layers to produce output
        return self.output(self.combined_fc(combined))
