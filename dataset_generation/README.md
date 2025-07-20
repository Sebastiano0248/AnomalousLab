# Dataset Generation Study

This study focuses on the generation of synthetic trajectory datasets for anomalous diffusion research. The goal is to create comprehensive datasets that can be used to train and validate machine learning models for various types of anomalous diffusion processes.

## Research Objectives

- **Synthetic Data Generation**: Create realistic trajectory datasets representing different types of anomalous diffusion processes (Brownian motion, fractional Brownian motion, continuous-time random walks, Lévy walks, etc.)
- **Data Quality Assessment**: Ensure the generated datasets accurately represent the theoretical properties of each diffusion type
- **Scalability Testing**: Develop efficient methods for generating large-scale datasets suitable for deep learning applications
- **Validation Framework**: Implement validation procedures to verify the statistical properties of generated trajectories

## Methodology

The study employs the [AnomalousLib](https://pypi.org/project/AnomalousLib/) library as the foundation for generating synthetic trajectories. Different trajectory types are generated with varying parameters to create diverse training datasets.

### Key Components

- **DatasetGeneratorTest.py**: Python script for testing dataset generation functionality and validating output quality
- **GenerateDatasets.ipynb**: Main notebook for systematic dataset generation with different parameters and trajectory types
- **generate_datasets.ipynb**: Additional notebook for specific dataset generation experiments

## Expected Outcomes

- High-quality synthetic datasets for training anomalous diffusion models
- Validated generation pipelines that can be reused for future research
- Documentation of optimal parameters for different trajectory types
- Benchmarking data for model comparison studies

## Usage

To generate datasets, run the Jupyter notebooks in the following order:
1. Start with `GenerateDatasets.ipynb` for the main generation pipeline
2. Use `generate_datasets.ipynb` for specific experiments
3. Validate results using `DatasetGeneratorTest.py`

## Dependencies

- AnomalousLib
- NumPy
- Pandas
- Matplotlib (for visualization)
- Jupyter Notebook
