"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.14
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Device configuration
device = torch.device('mps')

def split_data(model_input_table: pd.DataFrame, parameters: Dict) -> torch.tensor:
    print(f"Test size: {parameters['test_size']}, type: {type(parameters['test_size'])}")
    print(f"Random state: {parameters['random_state']}, type: {type(parameters['random_state'])}")

    # Split data into features and target
    X = model_input_table[model_input_table.columns[:-1]].values  # Assuming last column is the target
    y = model_input_table[model_input_table.columns[-1]].values
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = parameters["test_size"], random_state = parameters["random_state"])
    # Initialize StandardScaler
    scaler = StandardScaler()
    # Fit on training data
    scaler.fit(X_train)
    # Transform both training and testing data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Ensure y_train and y_test are in the correct format
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.astype(np.float32))
    X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32))
    y_test_tensor = torch.tensor(y_test.astype(np.float32))

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def train_model() -> None:
    pass

def evaluate_model() -> None:
    pass