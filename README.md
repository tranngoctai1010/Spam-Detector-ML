# AutoML Pipeline
A system to automate machine learning tasks like classification, regression, and clustering.

## Why this project?
- Automatically trains and evaluates models (Logistic Regression, XGBoost, etc.).
- Supports multiple datasets and tasks.
- Deployable with Docker for real-world use.

## Quick Start
1. Install: `pip install -r requirements.txt`
2. Run example: `python examples/auto_ml_classification.py`
3. Check results in `logs/src.log`

## Results
- Achieved 95% accuracy on Iris dataset with Random Forest.
- See `results/` for detailed metrics and plots.

## Tech Stack
Python, scikit-learn, XGBoost, LightGBM, Docker, GitHub Actions