# Financial Fraud Detection

## Project Overview

This project looks at a large financial transaction dataset and builds a machine learning model to detect fraudulent transactions. The workflow adheres to the analytics pipeline:

1. Exploratory data analysis to understand fraud patterns
2. Cleaning and preprocessing to prepare data ready for modeling
3. Feature engineering based on transaction and balance behavior
4. Random Forest modeling and evaluation
5. Interpretation of model performance and feature importance

The main goal is not only to predict fraud, but to find which transaction behaviors are most associated with fraudulent activity.

## Dataset

The raw dataset has **6,362,620 transactions** and **11 columns** describing simulated financial activity. Each row represents one transaction and includes transaction type, amount, sender and receiver balances, and a fraud label.

The raw CSV is not included in this repository because it is too large for GitHub. To rerun the notebooks from the beginning, place the source file at:

```text
data/raw/PS_20174392719_1491204439457_log.csv
```

Target variable:

- `isFraud`: whether the transaction is fraudulent

Important original features:

- `type`: transaction type
- `amount`: transaction amount
- `oldbalanceOrg`, `newbalanceOrig`: sender balance before and after transaction
- `oldbalanceDest`, `newbalanceDest`: receiver balance before and after transaction
- `isFlaggedFraud`: system flag included in the dataset

The dataset is highly imbalanced:

| Class | Count | Proportion |
| --- | ---: | ---: |
| Non-fraud | 6,354,407 | 99.87% |
| Fraud | 8,213 | 0.13% |

This imbalance is important because accuracy alone would be misleading. Precision, recall, F1 score, and ROC-AUC are more useful for evaluating fraud detection.

## Repository Structure

```text
financial-fraud-detection/
|-- data/
|   |-- raw/                  # Raw transaction dataset, ignored by git
|   `-- processed/            # Train, validation, and test CSV splits
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_cleaning_preprocessing.ipynb
|   `-- 03_modeling.ipynb
|-- requirements.txt
|-- README.md
```

The notebooks are where the bulk of the analysis occurs and are organized in the order of the analysis from top to bottom.

## Exploratory Data Analysis

The goal of EDA is to understand where fraud occurs, how fraud differs from non-fraud transactions, and what patterns could be useful for modeling.

### Key Findings

**Fraud is rare.** Fraudulent transactions make up only about **0.13%** of the full dataset. This confirms that the project is imbalanced.

**Fraud is concentrated in specific transaction types.** Fraud appears in `TRANSFER` and `CASH_OUT` transactions. No fraud was found in `PAYMENT`, `CASH_IN`, or `DEBIT` transaction types.

Fraud rate by transaction type:

| Transaction Type | Fraud Rate |
| --- | ---: |
| TRANSFER | 0.7688% |
| CASH_OUT | 0.1840% |
| CASH_IN | 0.0000% |
| DEBIT | 0.0000% |
| PAYMENT | 0.0000% |

**Fraudulent transactions tend to involve larger amounts.** The average fraud amount was much higher than the average non-fraud amount:

| Class | Average Amount |
| --- | ---: |
| Non-fraud | 178,197 |
| Fraud | 1,467,967 |

However, the distributions still overlap, so transaction amount alone is not enough to reliably separate fraud from non-fraud.

**Balance behavior provides stronger signals.** Balance-related features showed important differences between fraudulent and non-fraudulent transactions. Two engineered balance consistency features were especially useful:

```text
orig_balance_error = oldbalanceOrg - amount - newbalanceOrig
dest_balance_error = oldbalanceDest + amount - newbalanceDest
```

These features measure whether the sender and receiver balances changed as expected after a transaction.

Important balance observations:

- Fraudulent transactions often had more consistent origin balance movement.
- Destination balance errors were larger for fraudulent transactions.
- Balance-error features showed stronger modeling value than raw balance values alone.

**Fraud depends on feature combinations.** The correlation heatmap showed that individual features had weak linear correlation with fraud. This suggested that fraud detection would probably depend on combinations of transaction type, amount, and balance behavior rather than one single feature.

## Cleaning and Preprocessing

The preprocessing notebook prepared the dataset for modeling with a focus on keeping the model practical and analytically aligned with the EDA findings.

### Cleaning Steps

- Checked for missing values
- Checked for duplicate rows
- Filtered the modeling dataset to `TRANSFER` and `CASH_OUT` transactions
- Removed direct account identifiers:
  - `nameOrig`
  - `nameDest`
- Converted transaction type into binary indicators
- Created balance-error features
- Created a log-transformed amount feature
- Sampled non-fraud transactions to reduce runtime while preserving all fraud cases
- Split the processed data into train, validation, and test sets

The raw dataset had:

- **0 missing values**
- **0 duplicate rows**

### Feature Engineering

Created features:

| Feature | Purpose |
| --- | --- |
| `orig_balance_error` | Measures whether sender balance changed as expected |
| `dest_balance_error` | Measures whether receiver balance changed as expected |
| `orig_error_flag` | Flags sender balance inconsistencies |
| `dest_error_flag` | Flags receiver balance inconsistencies |
| `is_transfer` | Identifies transfer transactions |
| `is_cash_out` | Identifies cash-out transactions |
| `log_amount` | Reduces skew in transaction amount |

### Modeling Dataset

Because the full dataset is very large and highly imbalanced, the final modeling dataset included:

- All **8,213 fraud** transactions
- A random sample of **50,000 non-fraud** transactions

Final modeling dataset:

| Class | Count | Proportion |
| --- | ---: | ---: |
| Non-fraud | 50,000 | 85.89% |
| Fraud | 8,213 | 14.11% |

This made the project computationally manageable while preserving every fraud example.

### Data Splits

The processed dataset was split using stratified train, validation, and test sets.

| Split | Rows | Features | Fraud Rate |
| --- | ---: | ---: | ---: |
| Train | 40,749 | 14 | 14.11% |
| Validation | 8,732 | 14 | 14.11% |
| Test | 8,732 | 14 | 14.11% |

## Modeling

The project used the **Random Forest Classifier** because it can capture non-linear relationships and feature interactions, which matches EDA insight that fraud was not explained by one variable alone.

Final model parameters:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

A second, more complex Random Forest model was also tested. It performed nearly the same as the first model, so the simpler model was selected as the final model.

## Model Performance

### Validation Results

| Metric | Score |
| --- | ---: |
| Precision | 0.9992 |
| Recall | 0.9968 |
| F1 Score | 0.9980 |
| ROC-AUC | 0.9990 |

### Test Results

| Metric | Score |
| --- | ---: |
| Precision | 1.0000 |
| Recall | 0.9984 |
| F1 Score | 0.9992 |
| ROC-AUC | 0.9993 |

The model performed extremely well on the sampled test set. Because fraud detection has high costs for missed fraud, recall is especially important. The final model detected almost all fraud cases while also maintaining very high precision.

## Feature Importance

The most important model features were strongly tied to balance behavior.

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `orig_balance_error` | 0.3217 |
| 2 | `orig_error_flag` | 0.2600 |
| 3 | `newbalanceOrig` | 0.1014 |
| 4 | `oldbalanceOrg` | 0.0879 |
| 5 | `newbalanceDest` | 0.0669 |
| 6 | `dest_balance_error` | 0.0347 |
| 7 | `step` | 0.0310 |
| 8 | `amount` | 0.0256 |
| 9 | `oldbalanceDest` | 0.0242 |
| 10 | `dest_error_flag` | 0.0168 |

This supports the EDA conclusion that fraud is best identified through balance movement patterns and engineered balance consistency features.

## Main Takeaways

- Fraud is extremely rare in the full dataset, making this an imbalanced classification problem.
- Fraud occurs only in `TRANSFER` and `CASH_OUT` transactions in this dataset.
- Fraudulent transactions tend to have higher amounts, but amount alone is not enough for detection.
- Engineered balance-error features were the strongest predictors.
- A Random Forest model captured the interaction between transaction type, amount, and balance behavior effectively.
- The simpler Random Forest was selected because it performed nearly identically to the more complex model.

## Project Conclusion

This project shows that financial fraud detection benefits from combining domain-driven feature engineering with machine learning. The most useful signals came from whether transaction balances changed in expected ways, especially for `TRANSFER` and `CASH_OUT` transactions. The final Random Forest model achieved strong fraud-detection performance and confirmed that engineered balance features were more informative than transaction amount alone.
