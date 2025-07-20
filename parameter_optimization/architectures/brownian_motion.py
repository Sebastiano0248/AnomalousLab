"""
Arquitecturas Específicas para BrownianMotion - Todos los Parámetros
===================================================================

Arquitecturas CNN optimizadas específicamente para predecir cada parámetro
de las trayectorias de Brownian Motion.
"""

import torch
import torch.nn as nn
from .base_architecture import BaseParameterArchitecture

class BMDeltaTArchitecture(BaseParameterArchitecture):
    """
    Arquitectura específica para predecir el parámetro delta_t en Brownian Motion.
    
    Delta_t representa el paso temporal en la simulación de Brownian Motion.
    Afecta la resolución temporal y la suavidad de la trayectoria.
    """
    
    def __init__(self, input_channels=2):
        super().__init__(input_channels)
        self.trajectory_type = "BrownianMotion"
        self.parameter_name = "delta_t"
        self.architecture_info = {
            "description": "CNN optimizada para parámetro delta_t en Brownian Motion",
            "characteristics": [
                "Detección de resolución temporal",
                "Análisis de suavidad de trayectoria",
                "Arquitectura enfocada en patrones temporales"
            ],
            "optimal_range": "(0.1, 5.0)",
            "complexity": "medium"
        }
    
    def create_model(self) -> nn.Module:
        """
        Crea el modelo CNN específico para delta_t en Brownian Motion.
        
        Enfocado en detectar patrones temporales y resolución.
        """
        
        class BMDeltaTCNN(nn.Module):
            def __init__(self, input_channels):
                super().__init__()
                
                # Arquitectura enfocada en patrones temporales
                self.conv_layers = nn.Sequential(
                    # Bloque 1: Detección de resolución temporal
                    nn.Conv1d(input_channels, 48, kernel_size=7, padding=3),
                    nn.BatchNorm1d(48),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 2: Análisis de suavidad
                    nn.Conv1d(48, 96, kernel_size=5, padding=2),
                    nn.BatchNorm1d(96),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 3: Características finas
                    nn.Conv1d(96, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(6),
                )
                
                # Red densa para regresión
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128 * 6, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        return BMDeltaTCNN(self.input_channels)
    
    def get_training_params(self) -> dict:
        """
        Parámetros de entrenamiento optimizados para delta_t en Brownian Motion.
        """
        return {
            "epochs": 45,
            "batch_size": 24,
            "learning_rate": 0.0001,
            "weight_decay": 0.0008,
            "scheduler": "ReduceLROnPlateau",
            "patience": 8,
            "factor": 0.7,
            "optimizer": "Adam"
        }
    
    def get_data_params(self) -> dict:
        """
        Parámetros óptimos para generación de datos BM-delta_t.
        """
        return {
            "num_particles": 1200,
            "num_steps": 120,
            "delta_t_range": (0.1, 5.0),
            "trajectory_length": 2.5,
            "sampling_strategy": "uniform"
        }


class BMSigmaArchitecture(BaseParameterArchitecture):
    """
    Arquitectura específica para predecir el parámetro sigma en Brownian Motion.
    
    Sigma representa la intensidad del ruido/difusión en Brownian Motion.
    Controla la amplitud de las fluctuaciones.
    """
    
    def __init__(self, input_channels=2):
        super().__init__(input_channels)
        self.trajectory_type = "BrownianMotion"
        self.parameter_name = "sigma"
        self.architecture_info = {
            "description": "CNN optimizada para parámetro sigma en Brownian Motion",
            "characteristics": [
                "Detección de amplitud de fluctuaciones",
                "Análisis de varianza de trayectoria",
                "Arquitectura enfocada en intensidad de ruido"
            ],
            "optimal_range": "(0.1, 3.0)",
            "complexity": "medium"
        }
    
    def create_model(self) -> nn.Module:
        """
        Crea el modelo CNN específico para sigma en Brownian Motion.
        
        Enfocado en detectar la amplitud e intensidad del ruido.
        """
        
        class BMSigmaCNN(nn.Module):
            def __init__(self, input_channels):
                super().__init__()
                
                # Arquitectura enfocada en amplitud y varianza
                self.conv_layers = nn.Sequential(
                    # Bloque 1: Detección de amplitud
                    nn.Conv1d(input_channels, 56, kernel_size=9, padding=4),
                    nn.BatchNorm1d(56),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 2: Análisis de varianza
                    nn.Conv1d(56, 112, kernel_size=7, padding=3),
                    nn.BatchNorm1d(112),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 3: Patrones de fluctuación
                    nn.Conv1d(112, 168, kernel_size=5, padding=2),
                    nn.BatchNorm1d(168),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 4: Características finas de ruido
                    nn.Conv1d(168, 224, kernel_size=3, padding=1),
                    nn.BatchNorm1d(224),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(5),
                )
                
                # Red densa para regresión de sigma
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(224 * 5, 512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        return BMSigmaCNN(self.input_channels)
    
    def get_training_params(self) -> dict:
        """
        Parámetros de entrenamiento optimizados para sigma en Brownian Motion.
        """
        return {
            "epochs": 55,
            "batch_size": 20,
            "learning_rate": 0.00008,
            "weight_decay": 0.0012,
            "scheduler": "ReduceLROnPlateau",
            "patience": 10,
            "factor": 0.6,
            "optimizer": "Adam"
        }
    
    def get_data_params(self) -> dict:
        """
        Parámetros óptimos para generación de datos BM-sigma.
        """
        return {
            "num_particles": 1400,
            "num_steps": 140,
            "sigma_range": (0.1, 3.0),
            "trajectory_length": 3.0,
            "sampling_strategy": "log_uniform"  # Mejor para rangos de escala
        }
