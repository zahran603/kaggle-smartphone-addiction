# Smartphone Addiction Prediction (Kaggle S6E8)

Predicting smartphone addiction probability using tabular user behavioral data and LightGBM.

* Competition: Kaggle Playground Series s6e8
* Metric: ROC-AUC
* Target: addicted_label

---

## Workflow

1. Exploratory Data Analysis (EDA)
   * Target class distribution and proportion[cite: 1]
   * Feature correlation heatmap[cite: 1]
   * Feature distributions by target class (KDE plots)[cite: 1]
   * Train vs. Test data drift verification

2. Feature Engineering
   * Screen time shares (social, gaming, productive)
   * Work-life ratios (leisure-to-productive, screen-to-sleep)
   * Usage intensity (opens per hour, notifications per open, weekend surge)

3. Modeling & Validation
   * Model: LightGBM (LGBMClassifier)
   * Validation: 5-Fold Stratified K-Fold
   * Output: submission.csv (test probability predictions)

---

## How to Run

1. Install requirements:
   ```bash
   pip install pandas numpy scikit-learn lightgbm matplotlib seaborn

   Open kaggle_prediction.ipynb and run all cells sequentially.