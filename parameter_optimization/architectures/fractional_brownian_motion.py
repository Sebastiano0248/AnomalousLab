"""
Specific Architecture for FractionalBrownianMotion - Hurst Parameter
==================================================================

CNN architecture specifically optimized for predicting the Hurst parameter
in Fractional Brownian Motion trajectories.
"""

import torch
import torch.nn as nn
from .base_architecture import BaseParameterArchitecture

class FBMHurstArchitecture(BaseParameterArchitecture):
    """
    Specific architecture for predicting the Hurst parameter in FBM.
    
    This architecture is optimized based on the specific characteristics
    of the Hurst parameter and the properties of FBM trajectories.
    """
    
    def __init__(self, input_channels=2):
        super().__init__(input_channels)
        self.trajectory_type = "FractionalBrownianMotion"
        self.parameter_name = "hurst"
        self.architecture_info = {
            "description": "CNN optimized for Hurst parameter in FBM",
            "characteristics": [
                "Focus on long-range correlation patterns",
                "Kernels optimized for persistence/anti-persistence detection", 
                "Adaptive pooling for different temporal scales"
            ],
            "optimal_range": "(0.1, 0.9)",
            "proven_mse": 0.0725
        }
    
    def create_model(self) -> nn.Module:
        """
        Creates the specific CNN model for Hurst in FBM.
        
        Architecture optimized based on:
        - Hurst parameter correlation characteristics
        - Empirical test results (MSE: 0.0725)
        - Balance between accuracy and speed
        """
        
        class FBMHurstCNN(nn.Module):
            def __init__(self, input_channels):
                super().__init__()
                
                # Convolutional layers for Hurst pattern detection
                self.conv_layers = nn.Sequential(
                    # First layer: detects local correlation patterns
                    nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Second layer: medium-term correlation patterns
                    nn.Conv1d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Third layer: persistence/anti-persistence features
                    nn.Conv1d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(4),  # Adaptive for different lengths
                )
                
                # Dense layers for Hurst parameter regression
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 4, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()  # Hurst ∈ (0,1)
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        return FBMHurstCNN(self.input_channels)
    
    def get_training_params(self) -> dict:
        """
        Optimized training parameters for Hurst in FBM.
        
        These parameters have been optimized based on:
        - Hurst parameter characteristics
        - Previous experimental results
        - Balance between convergence and overfitting
        """
        return {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "weight_decay": 0.001,
            "scheduler": "ReduceLROnPlateau",
            "patience": 10,
            "optimizer": "Adam"
        }
    
    def get_data_params(self) -> dict:
        """
        Optimal parameters for FBM-Hurst data generation.
        
        Optimized for:
        - Capturing relevant Hurst characteristics
        - Balance between quality and generation time
        - Sufficient variability for generalization
        """
        return {
            "num_particles": 1000,
            "num_steps": 100,
            "hurst_range": (0.1, 0.9),
            "trajectory_length": 1.0,
            "sampling_strategy": "uniform"
        }
