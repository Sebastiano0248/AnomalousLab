# Automated Trajectory Parameter Regression System

## Description

Automated system for training CNN regression models that predict specific parameters of different trajectory types. Each trajectory type and parameter combination uses its own optimized CNN architecture to achieve maximum prediction accuracy.

## Project Structure

```
parameter_optimization/
├── automated_trainer.py                          # Main training script
├── architectures/                                # Parameter-specific architectures
│   ├── base_architecture.py                      # Abstract parent class
│   ├── fractional_brownian_motion.py
│   ├── scaled_brownian_motion.py
│   ├── continuous_time_random_walk.py
│   ├── brownian_motion.py
│   ├── brownian_motion_all.py
│   ├── levy_walks.py
│   ├── levy_walks_all.py
│   ├── annealed_transit_time_model.py
│   ├── annealed_transit_time_model_all.py
│   └── __init__.py
├── best_models/                                   # Best trained models with metadata
│   ├── FractionalBrownianMotionHurstRegressor.pth
│   ├── FractionalBrownianMotionHurstRegressor_metadata.json
│   ├── ScaledBrownianMotionBetaRegressor.pth
│   ├── ScaledBrownianMotionBetaRegressor_metadata.json
│   ├── LevyWalksAlphaRegressor.pth
│   ├── LevyWalksAlphaRegressor_metadata.json
│   ├── LevyWalksBetaRegressor.pth
│   ├── LevyWalksSpeedRegressor.pth
│   ├── BrownianMotionDelta_tRegressor.pth
│   ├── BrownianMotionSigmaRegressor.pth
│   ├── ContinuousTimeRandomWalkAlphaRegressor.pth
│   └── AnnealedTransitTimeModelGamma_shapeRegressor.pth
├── optimization_results/                         # Optimization analysis
│   ├── modelo_optimization_summary.md            # Detailed optimization summary
│   ├── best_models_summary.csv                   # Performance metrics summary
│   └── train_best_models.py                      # Script for training best models
├── model_analysis.ipynb                          # Model analysis notebook
├── trajectory_model_optimization.ipynb           # Optimization analysis notebook
└── README.md                                     # This documentation
```

## System Capabilities

### automated_trainer.py

Main training script that provides:
- **Automated parameter-specific training**: Uses optimized architectures for each trajectory-parameter combination
- **Integrated hyperparameter optimization**: Learning rate, batch size, epochs, and weight decay automatically configured
- **Data generation optimization**: Number of particles and steps optimized per parameter type
- **Model persistence**: Saves trained models using AnomalousLib's structure with complete metadata
- **Validation and evaluation**: Built-in performance assessment during training

### architectures/

Parameter-specific CNN architectures directory:
- **base_architecture.py**: Abstract parent class defining the architecture interface
- **Trajectory-specific modules**: Optimized CNN architectures for each trajectory type and parameter
- **Training configuration**: Each architecture includes optimal hyperparameters and data generation settings
- **Multi-parameter support**: Some trajectories have dedicated architectures for multiple parameters

### best_models/

Repository of trained models:
- **Model files (.pth)**: PyTorch model weights for each trained trajectory-parameter combination
- **Metadata files (.json)**: Training configuration, architecture details, and performance metrics
- **Complete coverage**: Models available for all supported trajectory types and their parameters

### optimization_results/

Analysis and optimization documentation:
- **modelo_optimization_summary.md**: Comprehensive optimization methodology, results analysis, and performance comparison
- **best_models_summary.csv**: Tabular performance metrics for all trained models
- **train_best_models.py**: Script for reproducing the best performing models with optimal configurations

### Analysis Notebooks

- **model_analysis.ipynb**: Detailed model performance analysis and behavior evaluation
- **trajectory_model_optimization.ipynb**: Optimization process visualization and performance comparison graphs

## Supported Trajectory Types

The system supports regression for the following trajectory types and their parameters:

- **FractionalBrownianMotion**: hurst parameter
- **ScaledBrownianMotion**: beta parameter  
- **ContinuousTimeRandomWalk**: alpha parameter
- **BrownianMotion**: delta_t, sigma parameters
- **LevyWalks**: alpha, beta, speed parameters
- **AnnealedTransitTimeModel**: gamma_shape parameter

## Usage

### Basic Training

```bash
# Train a specific trajectory-parameter model
python automated_trainer.py --trajectory_type FractionalBrownianMotion --parameter hurst

# List available trajectory-parameter combinations
python automated_trainer.py --list_specific
```

### Advanced Configuration

```bash
# Custom training parameters
python automated_trainer.py --trajectory_type ScaledBrownianMotion --parameter beta --epochs 50 --batch_size 64 --lr 0.0005

# Custom dataset generation
python automated_trainer.py --trajectory_type BrownianMotion --parameter sigma --num_particles 5000 --num_steps 200
```

**Configurable parameters:**
- `--num_particles`: Number of trajectories to generate
- `--num_steps`: Steps per trajectory
- `--epochs`: Training epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate
- `--weight_decay`: Regularization

## Loading Trained Models

```python
from anomalouslib.models.model_handler import load_model

# Load any trained model
model = load_model('./best_models/FractionalBrownianMotionHurstRegressor.pth')

# Access model metadata
metadata = load_model('./best_models/FractionalBrownianMotionHurstRegressor_metadata.json')
```

## Architecture Philosophy

This system implements **parameter-specific optimization**:

- **Dedicated architectures**: Each trajectory-parameter combination has its own optimized CNN architecture
- **Specialized training**: Hyperparameters are optimized specifically for each regression task
- **Optimized data generation**: Number of particles and steps tuned per parameter type
- **No generic solutions**: Focus on specialized architectures for maximum performance

## Requirements

- Python 3.12.10+
- AnomalousLab virtual environment activated
- PyTorch 
- AnomalousLib library

## Results and Performance Analysis

For detailed performance metrics, optimization methodology, and results comparison, see:
- **`optimization_results/modelo_optimization_summary.md`**: Complete optimization analysis and recommendations
- **`optimization_results/best_models_summary.csv`**: Tabular performance data for all models
- **Analysis notebooks**: Visualization and detailed performance evaluation
