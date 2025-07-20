"""
Arquitectura Específica para AnnealedTransitTimeModel - Parámetro gamma_shape
============================================================================

Arquitectura CNN optimizada específicamente para predecir el parámetro gamma_shape
de las trayectorias de Annealed Transit Time Model.
"""

import torch
import torch.nn as nn
from .base_architecture import BaseParameterArchitecture

class ATTMGammaShapeArchitecture(BaseParameterArchitecture):
    """
    Arquitectura específica para predecir el parámetro gamma_shape en ATTM.
    
    Gamma_shape controla la forma de la distribución gamma que determina
    los coeficientes de difusión variables en el tiempo en ATTM.
    """
    
    def __init__(self, input_channels=2):
        super().__init__(input_channels)
        self.trajectory_type = "AnnealedTransitTimeModel"
        self.parameter_name = "gamma_shape"
        self.architecture_info = {
            "description": "CNN optimizada para parámetro gamma_shape en ATTM",
            "characteristics": [
                "Detección de variabilidad temporal en difusión",
                "Análisis de patrones heterogéneos",
                "Arquitectura robusta para procesos complejos"
            ],
            "optimal_range": "(0.5, 5.0)",
            "complexity": "high"
        }
    
    def create_model(self) -> nn.Module:
        """
        Crea el modelo CNN específico para gamma_shape en ATTM.
        
        Enfocado en detectar variabilidad temporal en coeficientes de difusión.
        """
        
        class ATTMGammaShapeCNN(nn.Module):
            def __init__(self, input_channels):
                super().__init__()
                
                # Arquitectura profunda para detectar variabilidad temporal
                self.conv_layers = nn.Sequential(
                    # Bloque 1: Detección de heterogeneidad temporal
                    nn.Conv1d(input_channels, 80, kernel_size=11, padding=5),
                    nn.BatchNorm1d(80),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 2: Patrones de difusión variable
                    nn.Conv1d(80, 160, kernel_size=9, padding=4),
                    nn.BatchNorm1d(160),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 3: Características de distribución gamma
                    nn.Conv1d(160, 240, kernel_size=7, padding=3),
                    nn.BatchNorm1d(240),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 4: Patrones complejos de ATTM
                    nn.Conv1d(240, 320, kernel_size=5, padding=2),
                    nn.BatchNorm1d(320),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 5: Características finas
                    nn.Conv1d(320, 400, kernel_size=3, padding=1),
                    nn.BatchNorm1d(400),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(6),
                )
                
                # Red densa robusta para regresión compleja
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(400 * 6, 1200),
                    nn.ReLU(),
                    nn.Dropout(0.6),
                    nn.Linear(1200, 300),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(300, 75),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(75, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        return ATTMGammaShapeCNN(self.input_channels)
    
    def get_training_params(self) -> dict:
        """
        Parámetros de entrenamiento optimizados para gamma_shape en ATTM.
        
        Requiere entrenamiento extenso debido a la complejidad del proceso.
        """
        return {
            "epochs": 90,
            "batch_size": 14,  # Batch pequeño para estabilidad
            "learning_rate": 0.00004,
            "weight_decay": 0.003,
            "scheduler": "ReduceLROnPlateau",
            "patience": 18,
            "factor": 0.3,
            "optimizer": "Adam"
        }
    
    def get_data_params(self) -> dict:
        """
        Parámetros óptimos para generación de datos ATTM-gamma_shape.
        """
        return {
            "num_particles": 2200,
            "num_steps": 160,
            "gamma_shape_range": (0.5, 5.0),
            "trajectory_length": 5.0,
            "sampling_strategy": "log_uniform"  # Mejor para rangos amplios
        }
