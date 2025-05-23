# Diabetes Prediction Using Machine Learning

## Objective
Predict diabetes diagnosis among women using various machine learning models on medical features such as glucose, insulin, BMI, and age.

## key Techniques
- Data cleaning (missing value imputation, normalization)
- Exploratory Data Analysis (EDA)
- PCA and LASSO feature selection
- Second-order feature engineering (squared and interaction terms)
- Classification models with 10-fold cross-validation

## Tools & Libraries
- R, caret, MASS, e1071, glmnet, rpart, pROC
- ggplot2, corrplot, GGally

##  Models Implemented
- Logistic Regression  
- Linear Discriminant Analysis (LDA)  
- Quadratic Discriminant Analysis (QDA)  
- Naïve Bayes  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Support Vector Machine (SVM)

## Evaluation Metrics
- Accuracy, AUC (ROC), confusion matrix
- Type I and Type II error analysis
- Train-Test Split and 10-fold Cross-Validation

##  Key Results
| Model                | Test Accuracy | Test AUC | CV Accuracy | CV AUC |
|---------------------|---------------|----------|-------------|--------|
| LDA                 | 76.5%         | 0.851    | 78.5%       | 0.846  |
| Logistic Regression | 77.4%         | 0.843    | 77.5%       | 0.832  |
| SVM (Radial)        | 73.5%         | 0.827    | 78.1%       | 0.821  |
| Decision Tree       | 77.8%         | 0.824    | 74.0%       | 0.707  |

## Files Included
- `code/diabetes_classification.R`: Full source code with preprocessing, modeling, and evaluation
- `report/final_project_report.pdf`: Full analysis and discussion
- Plots and model outputs

## Conclusion
This project demonstrates how well-crafted machine learning pipelines can help diagnose diabetes with strong accuracy and interpretability. LDA emerged as the most balanced model; SVM offered strong performance after hyperparameter tuning.

---

Feel free to fork, clone, or reach out if you'd like to collaborate!

Contact: [Nikshay Policepatel](mailto:nikshaypatels@gmail.com)

