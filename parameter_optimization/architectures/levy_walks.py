"""
Arquitecturas Específicas para LevyWalks - Todos los Parámetros
==============================================================

Arquitecturas CNN optimizadas específicamente para predecir cada parámetro
de las trayectorias de Lévy Walks.
"""

import torch
import torch.nn as nn
from .base_architecture import BaseParameterArchitecture

class LWAlphaArchitecture(BaseParameterArchitecture):
    """
    Arquitectura específica para predecir el parámetro alpha en Lévy Walks.
    
    Alpha controla la distribución de los tamaños de salto en Lévy Walks.
    Valores cercanos a 1 producen saltos más largos, valores cercanos a 2 
    se acercan al comportamiento Browniano.
    """
    
    def __init__(self, input_channels=2):
        super().__init__(input_channels)
        self.trajectory_type = "LevyWalks"
        self.parameter_name = "alpha"
        self.architecture_info = {
            "description": "CNN para parámetro Alpha en Levy Walks",
            "characteristics": [
                "Detección de distribución de saltos",
                "Análisis de patrones de movimiento Lévy",
                "Arquitectura robusta para variabilidad"
            ],
            "optimal_range": "(1.1, 2.0)",
            "complexity": "high"
        }
    
    def create_model(self) -> nn.Module:
        """
        Crea el modelo CNN específico para alpha en Lévy Walks.
        
        Enfocado en detectar patrones de distribución de saltos.
        """
        
        class LWAlphaCNN(nn.Module):
            def __init__(self, input_channels):
                super().__init__()
                
                # Arquitectura robusta para detectar distribuciones Lévy
                self.conv_layers = nn.Sequential(
                    # Bloque 1: Detección inicial de saltos
                    nn.Conv1d(input_channels, 64, kernel_size=11, padding=5),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 2: Patrones de distribución
                    nn.Conv1d(64, 128, kernel_size=9, padding=4),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 3: Características de Lévy
                    nn.Conv1d(128, 256, kernel_size=7, padding=3),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 4: Patrones complejos
                    nn.Conv1d(256, 384, kernel_size=5, padding=2),
                    nn.BatchNorm1d(384),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(8),
                )
                
                # Red densa para regresión de alpha
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(384 * 8, 768),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(768, 192),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(192, 48),
                    nn.ReLU(),
                    nn.Linear(48, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        return LWAlphaCNN(self.input_channels)
    
    def get_training_params(self) -> dict:
        """
        Parámetros de entrenamiento optimizados para alpha en Lévy Walks.
        """
        return {
            "epochs": 70,
            "batch_size": 20,
            "learning_rate": 0.00007,
            "weight_decay": 0.0018,
            "scheduler": "ReduceLROnPlateau",
            "patience": 12,
            "factor": 0.5,
            "optimizer": "Adam"
        }
    
    def get_data_params(self) -> dict:
        """
        Parámetros óptimos para generación de datos LW-alpha.
        """
        return {
            "num_particles": 1800,
            "num_steps": 130,
            "alpha_range": (1.1, 2.0),
            "trajectory_length": 4.0,
            "sampling_strategy": "balanced"
        }


class LWBetaArchitecture(BaseParameterArchitecture):
    """
    Arquitectura específica para predecir el parámetro beta en Lévy Walks.
    
    Beta controla la asimetría de la distribución de saltos en Lévy Walks.
    Valores positivos favorecen saltos en una dirección, valores negativos en la otra.
    """
    
    def __init__(self, input_channels=2):
        super().__init__(input_channels)
        self.trajectory_type = "LevyWalks"
        self.parameter_name = "beta"
        self.architecture_info = {
            "description": "CNN optimizada para parámetro beta en Lévy Walks",
            "characteristics": [
                "Detección de asimetría direccional",
                "Análisis de bias en saltos",
                "Arquitectura sensible a direccionalidad"
            ],
            "optimal_range": "(-1.0, 1.0)",
            "complexity": "high"
        }
    
    def create_model(self) -> nn.Module:
        """
        Crea el modelo CNN específico para beta en Lévy Walks.
        
        Enfocado en detectar asimetrías direccionales.
        """
        
        class LWBetaCNN(nn.Module):
            def __init__(self, input_channels):
                super().__init__()
                
                # Arquitectura especializada en asimetría
                self.conv_layers = nn.Sequential(
                    # Bloque 1: Detección de direccionalidad
                    nn.Conv1d(input_channels, 72, kernel_size=13, padding=6),
                    nn.BatchNorm1d(72),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 2: Patrones asimétricos
                    nn.Conv1d(72, 144, kernel_size=11, padding=5),
                    nn.BatchNorm1d(144),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 3: Bias direccional
                    nn.Conv1d(144, 216, kernel_size=9, padding=4),
                    nn.BatchNorm1d(216),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 4: Características finas de asimetría
                    nn.Conv1d(216, 288, kernel_size=7, padding=3),
                    nn.BatchNorm1d(288),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(7),
                )
                
                # Red densa para regresión de beta
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(288 * 7, 864),
                    nn.ReLU(),
                    nn.Dropout(0.6),
                    nn.Linear(864, 216),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(216, 54),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(54, 1),
                    nn.Sigmoid()  # Beta puede ser negativo, se escalará después
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        return LWBetaCNN(self.input_channels)
    
    def get_training_params(self) -> dict:
        """
        Parámetros de entrenamiento optimizados para beta en Lévy Walks.
        """
        return {
            "epochs": 85,
            "batch_size": 16,
            "learning_rate": 0.00005,
            "weight_decay": 0.0025,
            "scheduler": "ReduceLROnPlateau",
            "patience": 15,
            "factor": 0.4,
            "optimizer": "Adam"
        }
    
    def get_data_params(self) -> dict:
        """
        Parámetros óptimos para generación de datos LW-beta.
        """
        return {
            "num_particles": 2000,
            "num_steps": 150,
            "beta_range": (-1.0, 1.0),
            "trajectory_length": 4.5,
            "sampling_strategy": "balanced_symmetric"  # Importante para parámetros simétricos
        }


class LWSpeedArchitecture(BaseParameterArchitecture):
    """
    Arquitectura específica para predecir el parámetro speed en Lévy Walks.
    
    Speed controla la velocidad de movimiento en Lévy Walks.
    Afecta la magnitud de los desplazamientos sin cambiar la distribución.
    """
    
    def __init__(self, input_channels=2):
        super().__init__(input_channels)
        self.trajectory_type = "LevyWalks"
        self.parameter_name = "speed"
        self.architecture_info = {
            "description": "CNN optimizada para parámetro speed en Lévy Walks",
            "characteristics": [
                "Detección de magnitud de desplazamientos",
                "Análisis de velocidad de movimiento",
                "Arquitectura enfocada en escalas espaciales"
            ],
            "optimal_range": "(0.1, 5.0)",
            "complexity": "medium"
        }
    
    def create_model(self) -> nn.Module:
        """
        Crea el modelo CNN específico para speed en Lévy Walks.
        
        Enfocado en detectar la magnitud de los movimientos.
        """
        
        class LWSpeedCNN(nn.Module):
            def __init__(self, input_channels):
                super().__init__()
                
                # Arquitectura enfocada en magnitud de movimientos
                self.conv_layers = nn.Sequential(
                    # Bloque 1: Detección de escala espacial
                    nn.Conv1d(input_channels, 60, kernel_size=9, padding=4),
                    nn.BatchNorm1d(60),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 2: Magnitud de desplazamientos
                    nn.Conv1d(60, 120, kernel_size=7, padding=3),
                    nn.BatchNorm1d(120),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 3: Patrones de velocidad
                    nn.Conv1d(120, 180, kernel_size=5, padding=2),
                    nn.BatchNorm1d(180),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    
                    # Bloque 4: Características de escala
                    nn.Conv1d(180, 240, kernel_size=3, padding=1),
                    nn.BatchNorm1d(240),
                    nn.ReLU(),
                    nn.AdaptiveMaxPool1d(6),
                )
                
                # Red densa para regresión de speed
                self.fc_layers = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(240 * 6, 720),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(720, 180),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(180, 45),
                    nn.ReLU(),
                    nn.Linear(45, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = self.fc_layers(x)
                return x
        
        return LWSpeedCNN(self.input_channels)
    
    def get_training_params(self) -> dict:
        """
        Parámetros de entrenamiento optimizados para speed en Lévy Walks.
        """
        return {
            "epochs": 60,
            "batch_size": 22,
            "learning_rate": 0.0001,
            "weight_decay": 0.0015,
            "scheduler": "ReduceLROnPlateau",
            "patience": 10,
            "factor": 0.6,
            "optimizer": "Adam"
        }
    
    def get_data_params(self) -> dict:
        """
        Parámetros óptimos para generación de datos LW-speed.
        """
        return {
            "num_particles": 1600,
            "num_steps": 140,
            "speed_range": (0.1, 5.0),
            "trajectory_length": 3.5,
            "sampling_strategy": "log_uniform"  # Mejor para rangos de escala
        }
