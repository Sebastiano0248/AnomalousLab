# Trajectory Models Optimization Summary

## Introduction

This document summarizes the optimization and training process of regression models to predict specific parameters of different types of anomalous trajectories. The models are based on CNN architectures specifically optimized for each trajectory type and parameter.

## Optimization Methodology

### General Process
1. **Hyperparameter Search**: A systematic hyperparameter search process was used for each combination of trajectory type and parameter.
2. **Specific Architectures**: Each model uses a CNN architecture specifically optimized for the trajectory type and target parameter.
3. **Evaluation**: Models were evaluated using Mean Squared Error (MSE) as the primary metric.

### Tested Configurations
- **Learning Rates**: [0.001, 0.0005, 0.0001, 0.00005, 0.00015, 0.002]
- **Batch Sizes**: [16, 32, 64, 128]
- **Epochs**: [20, 25, 50, 100]
- **Weight Decay**: [1e-5, 1e-4, 1e-3, 5e-3]
- **Number of Particles**: [15000, 20000]
- **Steps per Trajectory**: [300, 500]

## Optimization Results

### Top 3 Best Models

#### 1. Fractional Brownian Motion - Hurst Parameter
- **MSE**: 0.0004 (Best global performance)
- **Optimal Configuration**:
  - Learning Rate: 5e-05
  - Batch Size: 64
  - Epochs: 25
  - Weight Decay: 0.0001
  - Particles: 15000
  - Steps: 500
- **Training Time**: 183.8 seconds
- **Architecture**: FBMHurstArchitecture (Specific for FBM)

#### 2. Scaled Brownian Motion - Beta Parameter
- **MSE**: 0.0005
- **Optimal Configuration**:
  - Learning Rate: 0.0001
  - Batch Size: 64
  - Epochs: 25
  - Weight Decay: 0.0001
  - Particles: 15000
  - Steps: 500
- **Training Time**: 33.2 seconds
- **Architecture**: SBMBetaArchitecture (Specific for SBM)

#### 3. Annealed Transit Time Model - Gamma Shape Parameter
- **MSE**: 0.0052
- **Optimal Configuration**:
  - Learning Rate: 0.001
  - Batch Size: 128
  - Epochs: 25
  - Weight Decay: 0.001
  - Particles: 20000
  - Steps: 500
- **Training Time**: 80.0 seconds
- **Architecture**: ATTMGammaShapeArchitecture (Specific for ATTM)

### Complete Results

| Trajectory Type | Parameter | MSE | Time (s) | Difficulty |
|-----------------|-----------|-----|----------|------------|
| FractionalBrownianMotion | hurst | 0.0004 | 183.8 | Very Easy |
| ScaledBrownianMotion | beta | 0.0005 | 33.2 | Very Easy |
| AnnealedTransitTimeModel | gamma_shape | 0.0052 | 80.0 | Easy |
| ContinuousTimeRandomWalk | alpha | 0.0207 | 58.1 | Moderate |
| BrownianMotion | sigma | 0.0247 | 56.5 | Moderate |
| LevyWalks | alpha | 0.0386 | 69.0 | Moderate |
| LevyWalks | speed | 0.0488 | 33.8 | Difficult |
| BrownianMotion | delta_t | 0.0609 | 51.6 | Difficult |
| LevyWalks | beta | 0.0807 | 42.8 | Very Difficult |

## Pattern Analysis

### Success Factors
1. **Specific Architectures**: Best results were obtained with CNN architectures specifically designed for each trajectory type.
2. **Moderate Learning Rates**: Best models used learning rates between 5e-05 and 0.001.
3. **Medium Batch Sizes**: Batch sizes of 64 were more effective than extreme sizes.
4. **Moderate Regularization**: Weight decay between 1e-4 and 1e-3 showed better results.

### Prediction Difficulty
- **Easiest parameters**: Hurst (FBM), Beta (SBM)
- **Most difficult parameters**: Beta (Lévy Walks), Delta_t (Brownian Motion)

### Computational Efficiency
- Fastest models: SBM-Beta (33.2s), LW-Speed (33.8s)
- Slowest models: FBM-Hurst (183.8s) - but with the best MSE

## New Model Structure

The trained models use the new AnomalousLib structure that includes:
- **Complete metadata**: Information about architecture, target parameter, and configuration
- **Compatibility**: Use of `save_model()` and `load_model()` from AnomalousLib
- **Specific architectures**: Each model uses an architecture optimized for its use case

## Generated Files

### Final Models (best_models/)
- `FractionalBrownianMotionHurstRegressor.pth`
- `ScaledBrownianMotionBetaRegressor.pth`  
- `LevyWalksAlphaRegressor.pth`

### Training Data
- Complete results available in timestamped CSV files
- Intermediate configurations saved for safety
- Detailed progress metrics

## Recommendations

1. **For Production**: Use the top 3 models (FBM-Hurst, SBM-Beta, ATTM-GammaShape)
2. **For Research**: Explore improvements in models with MSE > 0.05
3. **Future Optimization**: Focus on LevyWalks-Beta which showed the highest difficulty

## Last Update Date
July 19, 2025

---
*Automatically generated from anomalous trajectory parameter optimization results*
