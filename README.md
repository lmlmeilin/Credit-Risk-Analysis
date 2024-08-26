# Credit Risk Analysis involving Machine Learning Models

As part of a group project for the module Optimisation Methods in Business Analytics, my group members and I made use of Python programming language to predict whether a credit card holder will default on their payment for the next month. 

The dataset includes various demographic and financial attributes of the cardholders, which are used to build and evaluate several machine learning models.
Dataset: https://www.kaggle.com/search?q=credit+card+client

## Key Sections of the Project
### 1. Exploratory Data Analysis
From the dataset, statistical averages, outliers, and data distribution were identified and this influenced the feature selection for machine learning.

### 2. Data Preprocessing
At this stage, we handled the missing data, applied one-hot encoding to the categorical variables and decided how to handle outliers.

### 3. Feature Selection
Correlation matrix was used to identify the most relevant features. Forward selection algorithm was also used to further refined feature set to improve model performance. 14 features were then selected to be key predictors in the machine learning model.

### 4. Model Development
Next, 5 machine learning models were trained using the training data: 
- Logistic Regression
- K-Nearest Neighbors (KNN)
- XGBoost
- Artificial Neural Network (ANN)
- Support Vector Machines (SVM)

### 5. Model Evaluation
We used F1 Score, MSE, RMSE, ROC-AUC Score, and Accuracy scores as a benchmark to conduct model evaluation. As a result, XGBoost and ANN were identified as the best performing models.
The hyperparameters of the ANN and XGBoost model were then further tuned to achieve better model performance.

## Final Results
XGBoost achieved the highest accuracy and balanced performance across all metrics (F1 Score, MSE, RMSE, ROC-AUC Score, and Accuracy).
Although ANN is slightly lower in accuracy compared to XGBoost after feature reduction, ANN still performed well, highlighting its ability to handle complex data.





