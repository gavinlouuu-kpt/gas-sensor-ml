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
import optuna
from optuna.trial import TrialState

from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('mps')


def split_data(model_input_table: pd.DataFrame, parameters: Dict) -> torch.tensor:
    _clean_input = model_input_table.ffill()
    # convert to numpy array
    # _clean_input = _clean_input.values

    # Split data into features and target
    X = _clean_input[_clean_input.columns[:-1]].values  # Assuming last column is the target
    y = _clean_input[_clean_input.columns[-1]].values
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = parameters["test_size"], random_state = parameters["random_state"])
    
    # Further split to create a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size = parameters["val_size"], random_state = parameters["random_state"])

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


# Optuna managed model

class RNN(nn.Module):
    # def __init__(self, input_size, hidden_size, num_layers, num_classes):
    def __init__(self, trial, input_size, num_classes):
        super(RNN, self).__init__()

        # Optuna suggests the number of layers and hidden size
        self.num_layers = trial.suggest_int("num_layers", 1, 3)
        self.hidden_size = trial.suggest_int("hidden_size", 30, 100)

        # Optuna suggests the dropout ratio of each layer
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size, 
            self.hidden_size, 
            self.num_layers, 
            batch_first=True,
            dropout=(dropout_rate if self.num_layers > 1 else 0), 
            )
        self.fc = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out, _ = self.lstm(x, (h0,c0))  
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def define_model(trial, input_size, num_classes):
    model = RNN(trial, input_size, num_classes)
    return model

# Optuna managed training


def objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, parameters):
    model = define_model(trial, parameters['input_size'], parameters['num_classes']).to(device)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Load dataset to dataloader 
    train_dataset = TensorDataset(X_train_tensor.to(device), y_train_tensor.to(device))
    val_dataset = TensorDataset(X_val_tensor.to(device), y_val_tensor.to(device))
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=parameters['batch_size'], shuffle=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=parameters["batch_size"], shuffle=True)
    

    # Training of the model
    n_total_steps = len(train_loader)
    for epoch in range(parameters['num_epochs']):
        model.train()

        for i, (bins, target) in enumerate(train_loader):
            bins = bins.reshape(-1, parameters['sequence_length'], parameters['input_size']).to(device)
            target = target.squeeze().to(device)

            # Forward pass
            outputs = model(bins)
            # Example of reshaping/squeezing if applicable
            outputs = outputs.squeeze()  # Removes dimensions of size 1
            # outputs = outputs[:64]  # Adjust if you need to slice the outputs
            # target = target.squeeze().to(device)  # Add an extra dimension to match outputs
            loss = criterion(outputs, target)
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Don't calculate gradients
            total_loss = 0
            count = 0

            for bins, target in val_loader:  # Replace with your validation loader
                bins = bins.reshape(-1, parameters['sequence_length'], parameters['input_size']).to(device)
                target = target.squeeze().to(device)  # Add an extra dimension to match outputs
                
                outputs = model(bins)
                outputs = outputs.squeeze()

                loss = criterion(outputs, target)
                total_loss += loss.item()
                count += 1
        
        model.train() # Set the model back to training mode

        rmse = np.sqrt(total_loss / count)
        trial.report(rmse, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return rmse
            # print(f'Epoch [{epoch+1}/{num_epochs}], RMSE on validation data: {rmse}')

def study_model(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, parameters, study_options):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, 
                                           X_train_tensor, 
                                           y_train_tensor, 
                                           X_val_tensor, 
                                           y_val_tensor, 
                                           parameters), 
                                        #    n_trials=1, 
                                        #    timeout=600)
                                           n_trials=study_options['n_trials'], 
                                           timeout=study_options['timeout'])

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    return trial 

    