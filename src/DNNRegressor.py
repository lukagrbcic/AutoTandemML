import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from model_factory import ModelFactory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
class TorchDNNRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, input_size, output_size, hidden_layers=None, model_type='mlp',
                 dropout=0.0, batch_norm=False, activation='relu', 
                 epochs=10, batch_size=32, learning_rate=0.01, 
                 criterion='mse', optimizer='adam', validation_split=0.1,
                 early_stopping_patience=10, verbose=False, output_activation=None, 
                 input_scaler=None, output_scaler=None):
        
        self.model_type = model_type
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.model = None
        self.optimizer = optimizer
        self.criterion = criterion
        self.factory = ModelFactory()
        self.verbose = verbose
        self.output_activation = output_activation
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
    
    
    def fit(self, X, y):
        
        if self.verbose == True:
            print ('Using device:', device)
        
        self.model = self.factory.create_model(
                   self.model_type,
                   input_size=self.input_size,
                   hidden_layers=self.hidden_layers,
                   output_size=self.output_size,
                   dropout=self.dropout,
                   batch_norm=self.batch_norm,
                   activation=self.activation,
                   output_activation=self.output_activation
               )
        
        if self.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
        X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                          test_size=self.validation_split, random_state=42)
        
        
        """INPUT SCALERS"""
        
        if self.input_scaler == 'MinMax':
            self.sc_input = MinMaxScaler()
            X_train = self.sc_input.fit_transform(X_train)
            X_val = self.sc_input.transform(X_val)
            

        elif self.input_scaler == 'Standard':
            self.sc_input = StandardScaler()
            X_train = self.sc_input.fit_transform(X_train)
            X_val = self.sc_input.transform(X_val)

        """OUTPUT SCALERS"""

        if self.output_scaler == 'MinMax':
            self.sc_output = MinMaxScaler()
            y_train = self.sc_output.fit_transform(y_train)
            y_val = self.sc_output.transform(y_val)
            
        elif self.output_scaler == 'Standard':
            self.sc_output = StandardScaler()
            y_train = self.sc_output.fit_transform(y_train)
            y_val = self.sc_output.transform(y_val)

        

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if self.criterion == 'mse':
            self.criterion = nn.MSELoss()
        else: 
                    
            def rmse_loss(outputs, targets):
                return torch.sqrt(torch.mean((outputs - targets) ** 2))    
            self.criterion = rmse_loss
        
        best_val_loss = float('inf')
        patience = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            for batch_X, batch_y in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                               
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                                
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
                
            if self.verbose == True:
                print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    if self.verbose == True:
                        print("Early stopping triggered")
                    break
    
    def predict(self, X):
                
        if self.input_scaler != None:
            X = self.sc_input.transform(X)
    
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)             
            outputs = self.model(X_tensor)
            predictions = outputs.cpu().numpy()
                        
            if self.output_scaler != None:
                predictions = self.sc_output.inverse_transform(predictions)
            
       
        return predictions