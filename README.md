# AnomalousLab
**AnomalousLab** is an AI-driven research environment built on top of [AnomalousLib](https://pypi.org/project/AnomalousLib/), designed for model inference and experimentation in the field of anomalous diffusion.

Its goal is to serve as a collaborative research platform, allowing contributors to share tools, datasets, models, and results through a unified and reproducible workflow.

> ⚙️ Recommended Python version: Python 3.12.10 or higher.

## Installation

### Set Up a Virtual Environment
Creating a virtual environment is not mandatory, but it is highly recommended for a clean installation. To set it up:
```bash
python -m venv venv
```
Then activate it:
- On **Linux/macOS**:
    ```bash
    source venv/bin/activate
    ```
- On **Windows**:
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
> **💡 Note:** To deactivate the environment at any time (on both Linux and Windows), use:
> ```bash
> deactivate
> ```

### AnomalousLib installation
As indicated on the [PyPI page](https://pypi.org/project/AnomalousLib/), install ```AnomalousLib``` using:
```bash
pip install anomalouslib
```

## Repository Structure
The repository is organized into several main components, where each directory represents a specific research study focused on different aspects of anomalous diffusion. Each study directory contains its own README file that explains the specific research objectives, methodology, and findings of that particular investigation:
```nginx
AnomalousLab
│   # Main documentation and dependencies
├───README.md
├───requirements.txt
│   # Dataset generation tools and notebooks
├───dataset_generation/
│   ├───README.md                            # Study documentation
│   ├───DatasetGeneratorTest.py
│   ├───GenerateDatasets.ipynb
│   └───generate_datasets.ipynb
│   # Trajectory classification experiments
├───trajectory_type_classifier/
│   ├───README.md                            # Study documentation
│   ├───TrajectoryTypeClassifier.ipynb
│   ├───testing_model_for_each_traj_type.ipynb
│   └───TrajectoryTypeClassifier.pth          # Trained model
│   # Parameter optimization and model training
├───parameter_optimization/
│   ├───automated_trainer.py                  # Automated training pipeline
│   ├───model_analysis.ipynb                 # Model performance analysis
│   ├───prova_prediccio_mitja.ipynb          # Prediction testing
│   ├───README.md                            # Module-specific documentation
│   ├───architectures/                       # Model architectures
│   │   ├───base_architecture.py
│   │   ├───brownian_motion.py
│   │   ├───fractional_brownian_motion.py
│   │   ├───continuous_time_random_walk.py
│   │   ├───levy_walks.py
│   │   ├───annealed_transit_time_model.py
│   │   └───scaled_brownian_motion.py
│   ├───best_models/                         # Optimized trained models
│   │   ├───BrownianMotionDelta_tRegressor.pth
│   │   ├───BrownianMotionSigmaRegressor.pth
│   │   ├───FractionalBrownianMotionHurstRegressor.pth
│   │   ├───ContinuousTimeRandomWalkAlphaRegressor.pth
│   │   ├───LevyWalksAlphaRegressor.pth
│   │   ├───LevyWalksBetaRegressor.pth
│   │   ├───LevyWalksSpeedRegressor.pth
│   │   ├───AnnealedTransitTimeModelGamma_shapeRegressor.pth
│   │   ├───ScaledBrownianMotionBetaRegressor.pth
│   │   └───*_metadata.json                  # Model metadata files
│   └───optimization_results/                # Training optimization results
└───old (in order to delete)/               # Legacy files (to be removed)
```

## Jupyter Notebooks
This repository primarily uses Jupyter Notebooks as the main format for experimentation, documentation, and result visualization.

Jupyter Notebooks are ideal for this project because they:
- Allow **interactive development** and **step-by-step debugging**.
- Combine **code**, **results**, **visualizations**, and **explanations** in a single document.
- Facilitate **reproducible research** and make it easier for collaborators to follow and contribute to experiments.

### How to Install Jupyter
If you need it, here is a brief explanation of how to install it.

To work with the notebooks, first ensure your virtual environment is activated, then install Jupyter:

```bash
pip install notebook
```

Once installed, you can launch the Jupyter Notebook interface with:
```bash
jupyter notebook
```
This will open a local web interface in your browser where you can navigate to and run any ```.ipynb``` file in the repository.

> 🔁 If you're using VS Code, Jupyter notebooks can also be run directly within the editor with the Jupyter extension.