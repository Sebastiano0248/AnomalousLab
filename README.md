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
The following is the current ideal structure for the repository. It may evolve over time as research progresses and new needs arise:
```nginx
AnomalousLab
│   # Jupyter notebooks for research and experimentation
├───experiments
│   │   # Classifier model studies
│   ├───classifier_models
│   │   ├───TrajectoryTypeClassifier
│   │   └───RandomWalkTypeClassifier
│   │   # Regressor model studies
│   └───regressor_models
│       ├───RandomWalkStopRegressor
│       ├───BrownianMotionDeltaTRegressor
│       ├───BrownianMotionSigmaRegressor
│       ├───FractionalBrownianMotionHurstRegressor
│       ├───ContinuousTimeRandomWalkAlphaRegressor
│       ├───LevyWalkAlphaRegressor
│       ├───LevyWalkBetaRegressor
│       ├───LevyWalkSpeedRegressor
│       ├───AnnealedTransientTimeModelGammaShapeRegressor
│       └───ScaledBrownianMotionBetaRegressor
│   # Trained models
├───models
|   # Experiment outputs
└───results
    │   # Statistical summaries and post-processing
    ├───analysis/
    │   # Model predictions (raw or processed)
    ├───predictions/
    │   # Evaluation scores (e.g., MSE, R²)
    ├───metrics/
    │   # Plots and charts
    ├───visualizations/
    │   # Notebooks, PDFs, markdown summaries
    ├───reports/
    │   # Experiment and training logs
    └───logs/
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