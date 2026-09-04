# Abalone Age & Characteristics Prediction (Machine Learning)

## Objective
Conservation efforts have not been enough to prevent abalone populations from moving toward extinction. This project builds machine learning models that predict abalone characteristics from physical measurements — supporting both *commercial* use and *conservation monitoring* of the species. Developed as CA1 for the Machine Learning module (L7 Diploma in Data Analytics, CCT College Dublin), following the *CRISP-DM* framework.

Two modelling tasks were carried out on the same dataset:
1. *Classification* — predict Sex (M / F / I) from physical measurements.
2. *Regression* — predict a continuous feature, with *L1 (Lasso)* and *L2 (Ridge)* regularisation applied to Linear Regression.

## Dataset
- *Source:* Provided via CCT College Dublin (Moodle), an updated version of the classic Abalone dataset.
- *Size:* 4,180 rows × 9 columns.
- *Variables:*

| Variable | Type | Description | Units |
|---|---|---|---|
| Sex | Categorical | M (Male), F (Female), I (Infant) | — |
| Length | Continuous | Longest shell measurement | mm |
| Diameter | Continuous | Shell diameter, perpendicular to length | mm |
| Height | Continuous | Height with meat inside shell | mm |
| Whole_weight | Continuous | Weight of the whole abalone | g |
| Shucked_weight | Continuous | Weight of the meat only | g |
| Viscera_weight | Continuous | Weight of the gut (after bleeding) | g |
| Shell_weight | Continuous | Weight of the dried shell | g |
| Rings | Integer | Number of shell rings (age = Rings + 1.5 years) | count |

## Data Cleaning & Preparation
- *Missing values:* Height, Shucked_weight and Viscera_weight had a small number of nulls; replaced with the *median*, chosen over the mean due to high variance and outliers visible in boxplots.
- *Duplicates:* 3 duplicate rows detected and removed.
- *Encoding:* Sex (and other non-numeric columns) converted from categorical to numeric.
- *Scaling:* Features normalised (different units — mm, g, count — and wide value ranges would otherwise distort model performance).

## Key Visualisations
- Boxplots of Height, Shucked_weight and Viscera_weight — used to justify median imputation over mean.
- Correlation matrix — used to select a suitable regression target after Rings proved unsuitable.
- Confusion matrix (Logistic Regression) — for the Sex classification task.
- Coefficient comparison chart (Linear vs Ridge vs Lasso) — feature importance for the regression task.

## Modelling & Metrics

*Classification (target: Sex)* — KNN, Decision Tree, Logistic Regression, Random Forest; single train/test split, evaluated on Accuracy, Precision, Recall, F1-score, and confirmed with k-fold cross-validation:

| Model | Accuracy | Precision | Recall | F1-Score | CV Accuracy |
|---|---|---|---|---|---|
| K-Nearest Neighbours | 0.531 | 0.527 | 0.531 | 0.526 | 0.524 |
| Decision Tree | 0.493 | 0.490 | 0.493 | 0.491 | 0.496 |
| *Logistic Regression* | *0.589* | 0.576 | 0.589 | 0.577 | *0.553* |
| Random Forest | 0.577 | 0.568 | 0.577 | 0.571 | 0.538 |

*Regression (target: Shucked_weight)* — after the correlation matrix showed Rings was not a strong enough target, Shucked_weight was selected for its strong relationship with the other features. Linear Regression, Ridge (L2) and Lasso (L1) were trained via a pipeline:

| Model | MSE | MAE | R² |
|---|---|---|---|
| Linear | 0.0019 | 0.0237 | 0.9546 |
| *Ridge* | 0.0018 | 0.0244 | *0.9574* |
| Lasso | 0.0019 | 0.0238 | 0.9552 |

Whole_weight was the strongest predictor across all three models; Lasso did not zero out any of the 8 coefficients, indicating every feature carried some relevance.

## Results & Limitations
- *Classification* was the weaker of the two tasks — the best model (Logistic Regression) reached only ~59% accuracy (55% cross-validated). The confusion matrix shows most errors occur between the Male and Infant classes, which overlap heavily in physical measurements — sex is inherently difficult to determine from size alone.
- *Regression* performed strongly (R² ≈ 0.95–0.96 across all three models), confirmed by cross-validation (~0.96 mean accuracy for Lasso).
- *Deployment* was out of scope for this project — the models were built for analysis and monitoring purposes, not for production use.

## Business Conclusion
While not production-ready, the regression model enables precise monitoring of abalone characteristics from easily obtainable measurements — supporting a balance between conserving the species and allowing those who harvest them commercially to do so sustainably.

## Files in this repository

| File | Description |
|---|---|
| [CA1_ML_sba25076.ipynb](./CA1_ML_sba25076.ipynb) | Full analysis notebook — EDA, cleaning, classification and regression models, evaluation |
| [Data Dictionary - Abalone dataset.docx](<./Data Dictionary - Abalone dataset.docx>) | Description of each dataset variable |
| [Report_Crisp-DM_MachineLearning.pdf](./Report_Crisp-DM_MachineLearning.pdf) | Full written report, structured around the CRISP-DM framework |
| [abalone.csv](./abalone.csv) | Source dataset (4,180 rows) |
