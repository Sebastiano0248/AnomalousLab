import argparse
import os
import sys
from datetime import datetime
import json
import pickle
import inspect

from anomalouslib.models.model_handler import save_model, load_model, get_class_def_from_notebook
from anomalouslib.models.testing_utils import evaluate_continuous, predict_continuous
from anomalouslib.data.preprocessing import generate_classification_tensors
from anomalouslib.data.DatasetGenerator import DatasetGenerator
from anomalouslib.data.preprocessing import min_max_normalize
from anomalouslib.models.training_utils import train_model
from anomalouslib.constants import params
from sklearn.preprocessing import LabelEncoder

from architectures import (
    get_parameter_architecture,
    list_available_architectures,
    PARAMETER_ARCHITECTURES
)

from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import torch
import numpy as np

TRAJECTORY_GENERATORS = {
    "RandomWalk": "generate_RandomWalk",
    "BrownianMotion": "generate_BrownianMotion", 
    "FractionalBrownianMotion": "generate_FractionalBrownianMotion",
    "ContinuousTimeRandomWalk": "generate_ContinuousTimeRandomWalk",
    "LevyWalks": "generate_LevyWalks",
    "AnnealedTransitTimeModel": "generate_AnnealedTransitTimeModel",
    "ScaledBrownianMotion": "generate_ScaledBrownianMotion"
}

TRAJECTORY_PARAMETERS = {
    "RandomWalk": ["movement_type"],
    "BrownianMotion": ["delta_t", "sigma"],
    "FractionalBrownianMotion": ["hurst"],
    "ContinuousTimeRandomWalk": ["alpha"],
    "LevyWalks": ["alpha", "beta", "speed"],
    "AnnealedTransitTimeModel": ["gamma_shape"],
    "ScaledBrownianMotion": ["beta"]
}

def generate_dataset(trajectory_type, num_particles, num_steps):
    generator = DatasetGenerator(num_particules=num_particles)
    generator_method = getattr(generator, TRAJECTORY_GENERATORS[trajectory_type])
    data = generator_method(num_passos=num_steps, random_params=True)
    return data

def prepare_data(data, trajectory_type, parameter):
    X_list = []
    y_list = []
    
    for idx, row in data.iterrows():
        trajectory = row['coords']
        if hasattr(trajectory, 'T'):
            X_list.append(trajectory.T)
        else:
            X_list.append(trajectory)
        
        attributes = row['attributes']
        
        if parameter == "movement_type":
            continue
        elif parameter in attributes:
            param_value = attributes[parameter]
            if isinstance(param_value, list):
                param_value = param_value[idx] if idx < len(param_value) else param_value[0]
            y_list.append(param_value)
        else:
            continue
    
    X_tensor = torch.stack([torch.FloatTensor(x) for x in X_list])
    y_tensor = torch.FloatTensor(y_list).unsqueeze(1)
    
    param_range = (min(y_list), max(y_list))
    
    y_min, y_max = param_range
    if y_max > y_min:
        y_normalized = (y_tensor - y_min) / (y_max - y_min)
    else:
        y_normalized = y_tensor
    
    return X_tensor, y_normalized, param_range

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    indices = torch.randperm(n)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_regressor(model, train_loader, val_loader, epochs, lr, weight_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
    
    model.load_state_dict(best_model)
    return model

def evaluate_model(model, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        predictions = model(X_test)
        
        mse = F.mse_loss(predictions, y_test).item()

def save_trained_model(model, trajectory_type, parameter, model_params, param_range, save_dir, architecture_name, architecture_instance=None):
    os.makedirs(save_dir, exist_ok=True)
    
    model_name = f"{trajectory_type}{parameter.capitalize()}Regressor"
    
    if architecture_instance:
        try:
            model_source = architecture_instance.get_model_source_code()
        except:
            model_source = f"""
import torch
import torch.nn as nn

class {model_name}(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        
    def forward(self, x):
        pass
"""
    else:
        model_source = f"""
import torch
import torch.nn as nn

class {model_name}(nn.Module):
    def __init__(self, input_channels=2):
        super().__init__()
        
    def forward(self, x):
        pass
"""
    
    model_type = "trajectory_parameter_regressor"
    
    constructor_params = {
        "input_channels": model_params.get("input_channels", 2)
    }
    
    input_channels = constructor_params.get("input_channels", 2)
    
    save_model(
        model_instance=model,
        model_type=model_type,
        model_source=model_source,
        params=constructor_params,
        input_channels=input_channels,
        model_name=model_name,
        path=save_dir
    )
    
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
    metadata = {
        "trajectory_type": trajectory_type,
        "parameter": parameter,
        "parameter_range": param_range,
        "architecture": architecture_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "training_completed": True,
        "original_model_params": model_params
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    filepath = os.path.join(save_dir, f"{model_name}.pth")
    
    return filepath

def show_available_parameters():
    for traj_type, parameters in TRAJECTORY_PARAMETERS.items():
        print(f"{traj_type}: {parameters}")

def main():
    parser = argparse.ArgumentParser(description="Automated Trajectory Parameter Regressor Trainer")
    
    parser.add_argument("--trajectory_type", choices=list(TRAJECTORY_PARAMETERS.keys()))
    parser.add_argument("--parameter")
    parser.add_argument("--list_parameters", action="store_true")
    parser.add_argument("--list_specific", action="store_true")
    parser.add_argument("--num_particles", type=int, default=10000)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--save_dir", default="./trained_models")
    
    args = parser.parse_args()
    
    if args.list_parameters:
        show_available_parameters()
        return
    
    if args.list_specific:
        specific_archs = list_available_architectures()
        for traj_type, params in specific_archs.items():
            print(f"{traj_type}:")
            for param, arch_class in params.items():
                print(f"  {param} → {arch_class.__name__}")
        return
    
    if not args.trajectory_type or not args.parameter:
        parser.error("--trajectory_type and --parameter are required for training")
    
    if args.parameter not in TRAJECTORY_PARAMETERS[args.trajectory_type]:
        return
    
    if args.parameter == "movement_type":
        return
    
    architecture_instance = get_parameter_architecture(
        args.trajectory_type, args.parameter, input_channels=2
    )
    
    training_params = architecture_instance.get_training_params()
    data_params = architecture_instance.get_data_params()
    
    if args.num_particles == 10000:
        args.num_particles = data_params.get("num_particles", args.num_particles)
    if args.num_steps == 200:
        args.num_steps = data_params.get("num_steps", args.num_steps)
    
    if any([args.batch_size, args.epochs, args.lr, args.weight_decay]):
        if args.batch_size: training_params["batch_size"] = args.batch_size
        if args.epochs: training_params["epochs"] = args.epochs  
        if args.lr: training_params["lr"] = args.lr
        if args.weight_decay: training_params["weight_decay"] = args.weight_decay
    
    data = generate_dataset(args.trajectory_type, args.num_particles, args.num_steps)
    
    X_tensor, y_tensor, param_range = prepare_data(data, args.trajectory_type, args.parameter)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_tensor, y_tensor)
    
    batch_size = training_params.get("batch_size", 32)
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size
    )
    
    input_channels = 2
    
    model = architecture_instance.create_model()
    model_info = architecture_instance.get_info()
    
    model_params = {
        "input_channels": input_channels,
        "architecture": "specific",
        "specific_class": model.__class__.__name__,
        "trajectory_type": args.trajectory_type,
        "parameter": args.parameter
    }
    
    lr = training_params.get("lr") or training_params.get("learning_rate", 1e-4)
    epochs = training_params.get("epochs", 15)
    weight_decay = training_params.get("weight_decay", 1e-3)
    
    best_model = train_regressor(
        model, train_loader, val_loader, 
        epochs, lr, weight_decay
    )
    
    evaluate_model(best_model, X_test, y_test)
    
    model_path = save_trained_model(
        best_model, args.trajectory_type, args.parameter, 
        model_params, param_range, args.save_dir, "specific", 
        architecture_instance=architecture_instance
    )

if __name__ == "__main__":
    main()
