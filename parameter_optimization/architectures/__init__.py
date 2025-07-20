"""
Parameter-Specific Architectures
===============================

This module contains CNN architectures specifically designed and optimized
for each parameter of each trajectory type.
"""

from .base_architecture import BaseParameterArchitecture

# Import specific architectures (keeping existing ones that don't have conflicts)
from .fractional_brownian_motion import FBMHurstArchitecture
from .scaled_brownian_motion import SBMBetaArchitecture
from .continuous_time_random_walk import CTRWAlphaArchitecture

# Import comprehensive architectures (now with clean names)
from .brownian_motion import BMDeltaTArchitecture, BMSigmaArchitecture
from .levy_walks import LWAlphaArchitecture, LWBetaArchitecture, LWSpeedArchitecture
from .annealed_transit_time_model import ATTMGammaShapeArchitecture

# Complete registry of specific architectures by trajectory and parameter
PARAMETER_ARCHITECTURES = {
    "FractionalBrownianMotion": {
        "hurst": FBMHurstArchitecture
    },
    "ScaledBrownianMotion": {
        "beta": SBMBetaArchitecture
    },
    "ContinuousTimeRandomWalk": {
        "alpha": CTRWAlphaArchitecture
    },
    "BrownianMotion": {
        "delta_t": BMDeltaTArchitecture,
        "sigma": BMSigmaArchitecture
    },
    "LevyWalks": {
        "alpha": LWAlphaArchitecture,
        "beta": LWBetaArchitecture,
        "speed": LWSpeedArchitecture
    },
    "AnnealedTransitTimeModel": {
        "gamma_shape": ATTMGammaShapeArchitecture
    },
    "RandomWalk": {
        # Note: movement_type is categorical, not supported for regression yet
    }
}

def get_parameter_architecture(trajectory_type: str, parameter: str, input_channels: int = 2):
    """
    Gets the specific architecture for a given parameter.
    
    Args:
        trajectory_type: Type of trajectory
        parameter: Parameter name
        input_channels: Number of input channels (default: 2 for x,y coordinates)
        
    Returns:
        BaseParameterArchitecture: Instance of the specific architecture
        
    Raises:
        ValueError: If no specific architecture exists for the parameter
    """
    if trajectory_type not in PARAMETER_ARCHITECTURES:
        raise ValueError(f"No architectures available for trajectory type: {trajectory_type}")
    
    trajectory_params = PARAMETER_ARCHITECTURES[trajectory_type]
    
    if parameter not in trajectory_params:
        available_params = list(trajectory_params.keys())
        raise ValueError(f"No architecture for parameter '{parameter}' in {trajectory_type}. "
                        f"Available parameters: {available_params}")
    
    architecture_class = trajectory_params[parameter]
    return architecture_class(input_channels=input_channels)

def list_available_architectures():
    """
    Lists all available specific architectures by trajectory type and parameter.
    
    Returns:
        dict: Dictionary with trajectory types as keys and their available parameters
    """
    return PARAMETER_ARCHITECTURES

__all__ = [
    'BaseParameterArchitecture',
    'get_parameter_architecture', 
    'list_available_architectures',
    'PARAMETER_ARCHITECTURES',
    # Fractional Brownian Motion
    'FBMHurstArchitecture',
    # Scaled Brownian Motion
    'SBMBetaArchitecture', 
    # Continuous Time Random Walk
    'CTRWAlphaArchitecture',
    # Brownian Motion - All parameters
    'BMDeltaTArchitecture',
    'BMSigmaArchitecture',
    # Lévy Walks - All parameters
    'LWAlphaArchitecture',
    'LWBetaArchitecture',
    'LWSpeedArchitecture',
    # Annealed Transit Time Model - All parameters
    'ATTMGammaShapeArchitecture'
]
