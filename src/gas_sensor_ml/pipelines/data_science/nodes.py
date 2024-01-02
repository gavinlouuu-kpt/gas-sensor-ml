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
import io

from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('mps')

def _clean_NaN (X_dataset: pd.DataFrame, model_input_table: pd.DataFrame) -> pd.DataFrame:
    X_dataset_df = pd.DataFrame(X_dataset, columns=model_input_table.columns[:-1])
    # Fill NaN values with the mean of the column
    X_dataset_df.fillna(X_dataset_df.mean(), inplace=True)
    # Convert back to numpy arrays
    X_dataset = X_dataset_df.values
    return X_dataset

def split_data(model_input_table: pd.DataFrame, parameters: Dict) -> torch.tensor:
    # Split data into features and target
    X = model_input_table[model_input_table.columns[:-1]].values  # Assuming last column is the target
    y = model_input_table[model_input_table.columns[-1]].values
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = parameters["test_size"], random_state = parameters["random_state"])
    
    # Further split to create a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size = parameters["val_size"], random_state = parameters["random_state"])
    
    X_train = _clean_NaN(X_train, model_input_table)
    X_val = _clean_NaN(X_val, model_input_table)
    X_test = _clean_NaN(X_test, model_input_table)

    # Initialize StandardScaler
    scaler = StandardScaler()
    # Fit on training data
    scaler.fit(X_train)

    # Transform both training and testing data
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Ensure y_train and y_test are in the correct format
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_val, pd.Series):
        y_val = y_val.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.astype(np.float32))

    X_val_tensor = torch.tensor(X_val_scaled.astype(np.float32))
    y_val_tensor = torch.tensor(y_val.astype(np.float32))

    X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32))
    y_test_tensor = torch.tensor(y_test.astype(np.float32))

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        # out, _ = self.rnn(x, h0)  
        # or:
        out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out

def train_model(X_train_tensor: torch.tensor, y_train_tensor: torch.tensor, 
                X_val_tensor: torch.tensor, y_val_tensor: torch.tensor, 
                parameters: Dict) -> ():
    # Device configuration
    device = torch.device('mps')
    
    # Load dataset to dataloader 
    train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
    val_dataset = TensorDataset(X_val_tensor.to(device), y_val_tensor.to(device))
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=parameters['batch_size'], shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=parameters["batch_size"], shuffle=True)
    
    # initiate model and send to device 
    model = RNN(input_size=parameters["input_size"], hidden_size=parameters["hidden_size"], 
                num_layers=parameters["num_layers"], num_classes=parameters["num_classes"]).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])  

# Train the model
    n_total_steps = len(train_loader)
    for epoch in range(parameters['num_epochs']):
        for i, (bins, target) in enumerate(train_loader):  
            bins = bins.reshape(-1, parameters['sequence_length'], parameters['input_size']).to(device)
            target = target.to(device)
        
        # Forward pass
        outputs = model(bins)
        # Example of reshaping/squeezing if applicable
        outputs = outputs.squeeze()  # Removes dimensions of size 1
        outputs = outputs[:64]  # Adjust if you need to slice the outputs

        target = target.unsqueeze(1).to(device)  # Add an extra dimension to match outputs
        loss = criterion(outputs, target)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_epochs = parameters['num_epochs']
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    # Calculate RMSE at the end of each epoch
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Don't calculate gradients
            total_loss = 0
            count = 0
            for bins, target in val_loader: 
                bins = bins.reshape(-1, parameters['sequence_length'], parameters['input_size']).to(device)
                target = target.unsqueeze(1).to(device)  # Add an extra dimension to match outputs
                outputs = model(bins)
                loss = criterion(outputs, target)
                total_loss += loss.item()
                count += 1
            rmse = np.sqrt(total_loss / count)
            print(f'Epoch [{epoch+1}/{num_epochs}], RMSE on validation data: {rmse}')
        model.train()  # Set the model back to training mode
    # Save the model after training
    # model_path = '../data/06_models/model.pth'
    lstm_model = model.state_dict()
    
    return lstm_model

def evaluate_model(lstm_model, X_test_tensor: torch.tensor, y_test_tensor: torch.tensor, parameters: Dict):
    """
    Calculates and logs the Root Mean Squared Error
    Args:
        lstm_model: model state dict to initiate a model
        X_test: test features
        y_test: test target
    """
    device = torch.device('mps')
    # initialise model to do inference
    inf_model = RNN(input_size=parameters['input_size'], 
                  hidden_size=parameters['hidden_size'], 
                  num_layers=parameters['num_layers'], 
                  num_classes=parameters['num_classes'])
    inf_model = inf_model.to(device)
    inf_model.load_state_dict(lstm_model)
    inf_model.eval()    
    
    criterion = nn.MSELoss()

    test_dataset = TensorDataset(X_test_tensor.to(device), y_test_tensor.to(device))
    test_loader = DataLoader(dataset=test_dataset, batch_size=parameters['batch_size'], shuffle=True)

    print("Number of NaN values in test data:")
    print(pd.DataFrame(X_test_tensor.numpy()).isna().sum())
    print(pd.DataFrame(y_test_tensor.numpy()).isna().sum())



    with torch.no_grad():  # Don't calculate gradients
            total_loss = 0
            count = 0
            for bins, target in test_loader:  # Replace with your validation loader
                bins = bins.reshape(-1, parameters['sequence_length'], parameters['input_size']).to(device)
                target = target.unsqueeze(1).to(device)  # Add an extra dimension to match outputs
                outputs = inf_model(bins)
                loss = criterion(outputs, target)
                total_loss += loss.item()
                count += 1
            rmse = np.sqrt(total_loss / count)
            print(f'RMSE on test data: {rmse}')

            

    