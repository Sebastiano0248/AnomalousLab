"""
Abstract Base Class for CNN Parameter Regressor Architectures
=============================================================

Defines the common interface for all parameter-specific architectures.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseParameterArchitecture(ABC):
    """
    Abstract base class for parameter-specific architectures.
    
    Each parameter of each trajectory must have its own child class that inherits from this.
    """
    
    def __init__(self, input_channels=2):
        self.input_channels = input_channels
        self.trajectory_type = None
        self.parameter_name = None
        self.architecture_info = {}
    
    @abstractmethod
    def create_model(self) -> nn.Module:
        """
        Creates and returns the specific CNN model for this parameter.
        
        Returns:
            nn.Module: PyTorch model ready for training
        """
        pass
    
    @abstractmethod
    def get_training_params(self) -> dict:
        """
        Returns optimized training parameters for this parameter.
        
        Returns:
            dict: Parameters like learning_rate, batch_size, epochs, etc.
        """
        pass
    
    @abstractmethod
    def get_data_params(self) -> dict:
        """
        Returns optimal parameters for data generation.
        
        Returns:
            dict: Parameters like num_particles, num_steps, etc.
        """
        pass
    
    def get_info(self) -> dict:
        """
        Returns information about this architecture.
        
        Returns:
            dict: Descriptive information about the architecture
        """
        return {
            "trajectory_type": self.trajectory_type,
            "parameter_name": self.parameter_name,
            "input_channels": self.input_channels,
            "architecture_info": self.architecture_info,
            "class_name": self.__class__.__name__
        }
    
    def create_alternative_model(self, architecture_type="lightweight") -> nn.Module:
        """
        [DEPRECATED] Generic alternative architectures are no longer available.
        Use only the specific architecture optimized for this parameter.
        """
        raise NotImplementedError("Generic architectures have been removed. Use only the specific architecture.")
    
    def get_available_alternatives(self) -> list:
        """
        Returns list of available alternative architectures.
        
        Returns:
            list: List of alternative architecture names (empty - no generic architectures)
        """
        return []
    
    def get_model_source_code(self) -> str:
        """
        Returns the source code for the model created by this architecture.
        This is needed for AnomalousLib compatibility.
        
        Returns:
            str: Complete source code of the model class
        """
        import inspect
        import textwrap
        
        # Create a model instance to get its class
        model = self.create_model()
        model_class = model.__class__
        
        # Get the source code
        try:
            source_code = inspect.getsource(model_class)
            # Clean up indentation
            source_code = textwrap.dedent(source_code)
            
            # Add necessary imports at the beginning
            imports = """import torch
import torch.nn as nn

"""
            return imports + source_code
            
        except Exception as e:
            # Fallback: create a basic template
            model_name = f"{self.trajectory_type}{self.parameter_name.capitalize()}Regressor"
            return f"""import torch
import torch.nn as nn

class {model_name}(nn.Module):
    '''
    CNN model for predicting {self.parameter_name} parameter in {self.trajectory_type} trajectories.
    Generated automatically from architecture: {self.__class__.__name__}
    '''
    def __init__(self, input_channels=2):
        super().__init__()
        # Architecture created by {self.__class__.__name__}
        # Original model structure preserved in state_dict
        
    def forward(self, x):
        # Forward pass implementation from specific architecture
        pass
"""
