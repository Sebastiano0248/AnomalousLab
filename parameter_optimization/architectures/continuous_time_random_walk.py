"""
Arquitectura Específica para ContinuousTimeRandomWalk - Parámetro Alpha
======================================================================

Arquitectura CNN optimizada específicamente para predecir el parámetro Alpha
en trayectorias de Continuous Time Random Walk.
"""

import torch
import torch.nn as nn
from .base_architecture import BaseParameterArchitecture

class CTRWAlphaArchitecture(BaseParameterArchitecture):
    """
    Arquitectura específica para predecir el parámetro Alpha en CTRW.
    
    Esta arquitectura está optimizada para capturar las características
    de los tiempos de espera y saltos en CTRW.
    """
    
    def __init__(self, input_channels=2):
        super().__init__(input_channels)
        self.trajectory_type = "ContinuousTimeRandomWalk"
        self.parameter_name = "alpha"
        self.architecture_info = {
            "description": "CNN robusta para parámetro Alpha en CTRW",
            "characteristics": [
                "Detección de patrones de tiempos de espera",
                "Análisis de distribuciones de saltos",
                "Arquitectura profunda para complejidad de CTRW"
            ],
            "optimal_range": "(0.1, 2.0)",
            "complexity": "high"
        }
    
    def create_model(self) -> nn.Module:
        """
        Crea el modelo CNN específico para Alpha en CTRW.
        
        Arquitectura más profunda y robusta debido a la complejidad
        inherente de los procesos CTRW.
        """
        
        class CTRWAlphaCNN(nn.Module):
            def __init__(self, input_channels):
                super().__init__()
                
                # Arquitectura más profunda para capturar complejidad CTRW
                self.conv_layers = nn.Sequential(
                    # Bloque 1: Detección inicial de patrones
                    nn.Conv1d(input_channels, 64, kernel_size=9, padding=4),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 2: Patrones de tiempos de espera
                    nn.Conv1d(64, 128, kernel_size=7, padding=3),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 3: Características de saltos
                    nn.Conv1d(128, 256, kernel_size=5, padding=2),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 4: Patrones complejos de CTRW
                    nn.Conv1d(256, 512, kernel_size=3, padding=1),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(8),
                )
                
                # Red densa robusta para regresión compleja
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(512 * 8, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()  # Normalizar y escalar después
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        return CTRWAlphaCNN(self.input_channels)
    
    def get_training_params(self) -> dict:
        """
        Parámetros de entrenamiento optimizados para Alpha en CTRW.
        
        Requiere más épocas y regularización debido a la complejidad.
        """
        return {
            "epochs": 80,
            "batch_size": 16,  # Batch más pequeño por la complejidad
            "learning_rate": 0.00005,
            "weight_decay": 0.002,
            "scheduler": "ReduceLROnPlateau",
            "patience": 15,
            "factor": 0.5,
            "optimizer": "Adam"
        }
    
    def get_data_params(self) -> dict:
        """
        Parámetros óptimos para generación de datos CTRW-Alpha.
        
        Requiere más datos debido a la complejidad del proceso.
        """
        return {
            "num_particles": 1500,
            "num_steps": 150,
            "alpha_range": (0.1, 2.0),
            "trajectory_length": 3.0,
            "sampling_strategy": "balanced"  # Balanceado para diferentes regímenes
        }
