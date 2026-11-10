# Model Comparison Report

## 1. Objective
The objective of this report is to compare the performance of the supervised **XGBoost** classifier against the unsupervised **Autoencoder** model to determine the most effective approach for identifying target instances on the standardized test split.

## 2. Model Roles and Operational Architecture
* **XGBoost:** Functions as the primary production supervised classifier.
* **Autoencoder:** Operates as an experimental unsupervised anomaly detector.
* **Integration Strategy:** The Autoencoder is **not** replacing XGBoost; instead, it provides an additional anomaly/risk score as a supplementary signal.

## 3. Dataset and Identical Test Split
Both models were evaluated using an identical test split to ensure a fair, rigorous, and direct benchmark comparison.

## 4. XGBoost Metrics
The XGBoost model was evaluated using a classification threshold of $0.6$:
* **Accuracy:** $99.96\%$
* **Precision:** $94.94\%$
* **Recall:** $80.65\%$
* **F1 Score:** $87.21\%$
* **PR AUC:** $84.65\%$

## 5. Autoencoder Metrics
The Autoencoder model was evaluated using a reconstruction error threshold of $8.67$:
* **Precision:** $28.23\%$ (Validation: $33.61\%$)
* **Recall:** $47.30\%$ (Validation: $54.05\%$)
* **F1 Score:** $35.35\%$ (Validation: $41.45\%$)
* **PR AUC:** $32.21\%$
* **Confusion Matrix:**
  * True Negatives: $42,559$ | False Positives: $89$
  * False Negatives: $39$ | True Positives: $35$

## 6. Comparison Table

| Metric | XGBoost | Autoencoder |
| :--- | :--- | :--- |
| **Threshold** | $0.60$ | $8.67$ |
| **Precision** | $94.94\%$ | $28.23\%$ |
| **Recall** | $80.65\%$ | $47.30\%$ |
| **F1 Score** | $87.21\%$ | $35.35\%$ |
| **PR AUC** | $84.65\%$ | $32.21\%$ |
| **Accuracy** | $99.96\%$ | N/A |[cite: 1, 2]

## 7. Final Conclusion
The supervised **XGBoost** model significantly outperforms the unsupervised **Autoencoder** across all evaluation metrics, maintaining its role as the production classifier. The Autoencoder remains valuable as a secondary tool that provides an additional anomaly/risk score for monitoring and future ensemble experiments.