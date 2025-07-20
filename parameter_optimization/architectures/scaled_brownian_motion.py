"""
Specific Architecture for ScaledBrownianMotion - Beta Parameter
=============================================================

CNN architecture specifically optimized for predicting the Beta parameter
in Scaled Brownian Motion trajectories.
"""

import torch
import torch.nn as nn
from .base_architecture import BaseParameterArchitecture

class SBMBetaArchitecture(BaseParameterArchitecture):
    """
    Arquitectura específica para predecir el parámetro Beta en SBM.
    
    Esta arquitectura está optimizada para capturar las características
    de escalado temporal del parámetro Beta en SBM.
    """
    
    def __init__(self, input_channels=2):
        super().__init__(input_channels)
        self.trajectory_type = "ScaledBrownianMotion"
        self.parameter_name = "beta"
        self.architecture_info = {
            "description": "CNN optimizada para parámetro Beta en SBM",
            "characteristics": [
                "Enfoque en patrones de escalado temporal",
                "Detección de varianza creciente/decreciente",
                "Capas especializadas en tendencias no-lineales"
            ],
            "optimal_range": "(0.1, 1.9)",
            "complexity": "medium"
        }
    
    def create_model(self) -> nn.Module:
        """
        Crea el modelo CNN específico para Beta en SBM.
        
        Arquitectura optimizada para:
        - Detectar patrones de escalado temporal
        - Capturar varianza variable en el tiempo
        - Distinguir diferentes regímenes de Beta
        """
        
        class SBMBetaCNN(nn.Module):
            def __init__(self, input_channels):
                super().__init__()
                
                # Capas convolutivas para detectar patrones de escalado
                self.conv_layers = nn.Sequential(
                    # Primera capa: patrones locales de escalado
                    nn.Conv1d(input_channels, 48, kernel_size=7, padding=3),
                    nn.BatchNorm1d(48),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Segunda capa: tendencias de varianza
                    nn.Conv1d(48, 96, kernel_size=5, padding=2),
                    nn.BatchNorm1d(96),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Tercera capa: patrones de escalado a largo plazo
                    nn.Conv1d(96, 192, kernel_size=3, padding=1),
                    nn.BatchNorm1d(192),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(6),
                )
                
                # Capas densas para regresión del parámetro Beta
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(192 * 6, 384),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(384, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()  # Normalizar y luego escalar en post-procesamiento
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        return SBMBetaCNN(self.input_channels)
    
    def get_training_params(self) -> dict:
        """
        Parámetros de entrenamiento optimizados para Beta en SBM.
        """
        return {
            "epochs": 60,
            "batch_size": 24,
            "learning_rate": 0.00008,
            "weight_decay": 0.0015,
            "scheduler": "CosineAnnealingLR",
            "T_max": 30,
            "optimizer": "Adam"
        }
    
    def get_data_params(self) -> dict:
        """
        Parámetros óptimos para generación de datos SBM-Beta.
        """
        return {
            "num_particles": 1200,
            "num_steps": 120,
            "beta_range": (0.1, 1.9),
            "trajectory_length": 2.0,
            "sampling_strategy": "log_uniform"  # Mejor distribución para Beta
        }
