# Trajectory Type Classification Study

This study focuses on developing and evaluating machine learning models for classifying different types of anomalous diffusion trajectories. The primary goal is to create robust classifiers that can accurately distinguish between various diffusion processes based solely on trajectory data.

## Research Objectives

- **Multi-class Classification**: Develop models capable of classifying trajectories into different anomalous diffusion types (Brownian motion, fractional Brownian motion, continuous-time random walks, Lévy walks, etc.)
- **Model Architecture Optimization**: Explore different neural network architectures to find the most effective approach for trajectory classification
- **Performance Evaluation**: Comprehensive testing of model performance across different trajectory lengths, noise levels, and parameter ranges
- **Robustness Analysis**: Assess model performance under various real-world conditions and edge cases

## Methodology

The study employs convolutional neural networks (CNNs) and other deep learning architectures to learn distinctive features from trajectory data. The approach focuses on extracting spatial and temporal patterns that are characteristic of each diffusion type.

### Key Components

- **TrajectoryTypeClassifier.ipynb**: Main notebook containing the development, training, and evaluation of the trajectory classification model
- **testing_model_for_each_traj_type.ipynb**: Comprehensive testing notebook that evaluates model performance for each individual trajectory type
- **TrajectoryTypeClassifier.pth**: Trained model file containing the optimized neural network weights

## Model Architecture

The classifier uses a CNN-based architecture designed to:
- Extract spatial features from trajectory coordinates
- Capture temporal dependencies in the movement patterns
- Handle variable-length trajectories through appropriate preprocessing
- Provide probabilistic outputs for each trajectory type

## Evaluation Metrics

The study employs multiple evaluation metrics including:
- Classification accuracy
- Precision, recall, and F1-score for each class
- Confusion matrices
- ROC curves and AUC scores
- Cross-validation performance

## Expected Outcomes

- High-accuracy trajectory type classifier (>90% accuracy target)
- Comprehensive performance analysis across different trajectory types
- Insights into which trajectory features are most discriminative
- Robust model suitable for real-world applications

## Usage

To work with the trajectory classifier:
1. **Training**: Use `TrajectoryTypeClassifier.ipynb` to train new models or retrain existing ones
2. **Testing**: Run `testing_model_for_each_traj_type.ipynb` for detailed performance evaluation
3. **Inference**: Load the trained model from `TrajectoryTypeClassifier.pth` for classifying new trajectories

## Dependencies

- PyTorch
- AnomalousLib
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Future Work

- Extension to handle real experimental data
- Implementation of ensemble methods
- Development of uncertainty quantification
- Integration with parameter estimation models
