# radiomics-ml-pipeline
# Radiomics Machine Learning Pipeline

This repository contains the full implementation of a radiomics-based
machine-learning pipeline for prediction of clinical endpoints using 
multi-VOI features. The framework includes:

- Data cleaning, harmonization, and outlier handling
- Feature selection (Spearman filtering + LASSO)
- Stability selection based on cross-validation frequency
- Multiple machine-learning classifiers (LR, SVM, RF, KNN, XGBoost)
- Nested cross-validation
- External testing
- ROC, AUC confidence intervals, calibration curves, and DCA
- SHAP and permutation importance (for Random Forest & XGBoost)

## Folder Structure
See project tree in the repository.

radiomics-ml-pipeline/
├── data/
│   ├── raw/                   # Raw imaging data (not included)
│   ├── processed/             # Processed features and labels
│   └── splits/                # Train/validation/test splits
├── src/
│   ├── preprocessing/         # Data cleaning and harmonization
│   ├── feature_selection/     # Spearman + LASSO selection
│   ├── models/               # ML classifiers implementation
│   ├── evaluation/           # Metrics and visualization
│   └── utils/               # Helper functions
├── config/
│   ├── paths.yaml           # Data paths configuration
│   └── model_params.yaml    # Hyperparameter settings
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
└── run_pipeline.py         # Main execution script
