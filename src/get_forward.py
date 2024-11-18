import numpy as np
from model_factory import ModelFactory
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class forwardDNN:
    
    def __init__(self, X, y, hyperparameters, validation_split=0.1,
                 criterion='rmse', optimizer='adam', verbose=False, early_stopping_patience=10):
        
        self.X = X
        self.y = y
        self.hyperparameters = hyperparameters
        self.validation_split = validation_split
        self.criterion = criterion
        self.optimizer = optimizer
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
    
    def train_save(self):
        
        self.model = ModelFactory().create_model(model_type=self.hyperparameters['model_type'], 
                                                  input_size=np.shape(self.X)[1], 
                                                  output_size=np.shape(self.y)[1],
                                                  hidden_layers=self.hyperparameters['hidden_layers'],
                                                  dropout=self.hyperparameters['dropout'],
                                                  output_activation=self.hyperparameters['output_activation'],
                                                  batch_norm=self.hyperparameters['batch_norm'],
                                                  activation=self.hyperparameters['activation'])
        
        if self.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['learning_rate'])
            
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y,
                                                          test_size=self.validation_split, random_state=42)
        
        """INPUT SCALERS"""
        
        if self.hyperparameters['input_scaler'] == 'MinMax':
            self.sc_input = MinMaxScaler()
            X_train = self.sc_input.fit_transform(X_train)
            X_val = self.sc_input.transform(X_val)
            

        elif self.hyperparameters['input_scaler'] == 'Standard':
            self.sc_input = StandardScaler()
            X_train = self.sc_input.fit_transform(X_train)
            X_val = self.sc_input.transform(X_val)

        """OUTPUT SCALERS"""

        if self.hyperparameters['output_scaler'] == 'MinMax':
            self.sc_output = MinMaxScaler()
            y_train = self.sc_output.fit_transform(y_train)
            y_val = self.sc_output.transform(y_val)
            
        elif self.hyperparameters['output_scaler'] == 'Standard':
            self.sc_output = StandardScaler()
            y_train = self.sc_output.fit_transform(y_train)
            y_val = self.sc_output.transform(y_val)

    
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.hyperparameters['batch_size'], shuffle=False)
        

        if self.criterion == 'mse':
            self.criterion = nn.MSELoss()
        else: 
            def rmse_loss(outputs, targets):
                return torch.sqrt(torch.mean((outputs - targets) ** 2))  
            
            self.criterion = rmse_loss
        
        best_val_loss = float('inf')
        patience = 0
        
        epochs = self.hyperparameters['epochs']
        

        for epoch in range(epochs):
            self.model.train()
            epoch_training_loss = 0.0  # Initialize training loss for the epoch
            for batch_X, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)  # Correct argument order
                loss.backward()
                self.optimizer.step()
                epoch_training_loss += loss.item()
            
            average_training_loss = epoch_training_loss / len(train_dataloader)
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {average_training_loss}, Validation Loss: {val_loss.item()}")
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    if self.verbose:
                        print("Early stopping triggered")
                    break

        torch.save(self.model.state_dict(), 'forwardDNN.pth')
        
        
