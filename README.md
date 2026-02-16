# Blood‑Brain Barrier Permeability Prediction

This repository compares multiple machine learning approaches for predicting whether small molecules can permeate the blood–brain barrier (BBB). The project evaluates different input molecular representations[morgan/maccs fingerprints,molecular descriptors], model families[logistic regression,random forests,anns], and class-imbalance treatments[SMOTE,class-imbalance weighting], and compares the results of different approaches.

**Key Highlights**
- **Datasets:** B3DB classification and BBBP datasets (in `data/` folder).
- **Feature representations:** RDKit molecular descriptors, MACCS keys, Morgan fingerprints, and combinations.
- **Models evaluated:** Logistic Regression, Random Forest, XGBoost, ensemble methods (voting/stacking), and an ANN .
- **Class imbalance handling:** SMOTE, class weighting, and undersampling comparisons.
- **Evaluation:** Accuracy, precision/recall, F1, ROC AUC, Precision‑Recall curves, and model comparison visualizations saved to `figures/`.

**Notebooks & Code**
- **Traditional ML experiments:** [models/traditional_ml_methods.ipynb](models/traditional_ml_methods.ipynb)
- **Neural network experiments:** [models/ann.ipynb](models/ann.ipynb)
- **Feature extraction / helper scripts:** [models/dataset_analysis.py](models/dataset_analysis.py)

**Setup Instructions**
Create and activate a Python environment (conda recommended for RDKit):

```
conda create -n bbbp python=3.10 -y
conda activate bbbp
conda install -c conda-forge rdkit pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn -y
```

