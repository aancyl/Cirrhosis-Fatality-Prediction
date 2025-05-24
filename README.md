# Cirrhosis-Fatality-Prediction
![alt text](https://www.medstarhealth.org/-/media/project/mho/medstar/blogs/blogmedstar2021/thomas-liver-disease-desktop.jpg)

# Introduction
Accurately predicting patient outcomes is crucial for clinical decision-making and resource planning. In this project, we utilize patient data to build a predictive model that determines the survival status based on various clinical features.

The dataset includes medical information, presenting correlations between different variables and patient outcomes. By predicting whether a patient will survive, die, or undergo liver transplantation, healthcare providers can make more informed and timely interventions.

This problem is a heavily under-researched topic, and my work provides vauable insight and predictive modeling approach to adress this gap. This repository contains 2 Jupyter Notebooks that demonstrate a Deep Learning as well as a Machine Learning approach to predict patient survival outcomes based on clinical data. The primary objective is to classify patients into one of three categories:

0 = D (Death)

1 = C (Censored)

2 = CL (Censored due to Liver Transplantation)

# Dataset Information

This project utilizes two datasets: one obtained from an online source and the other from the UC Irvine Machine Learning Repository. The online dataset is used for training the model, while the UC Irvine dataset is used for testing.

Link to the UC Irvine Dataset for Cirrhosis Patient Survival Prediction : https://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1

# Methodology
## 1. Research / Understanding the problem statement.

* **Objective:** To gain industry spesific insights and a deeper understanding of the effect and the causes of cirrhosis.

I have spent a significant amount of time researching cirrhosis, including its causes, adverse effects, and progression. This understanding has helped me gain deeper insights into the data and its relevance to the disease.
  
## 2. Explooratory Data Analysis.

* **Ojbective:** Explore the features of the dataset and understand how they relate to the target variable.

Visualization techniques have been used to understand the impact of various indiviual features and the overall impact of the features combined with reference to the target variable. Performing this stage appropriately will provide useful knowledge for the further processes that will come into practice later.


## 3. Pre-Processing the Data.

* **Ojbective:** Clean the data impute missing data, remove outliers and imporove the predictive power of the model using esentail features. 

Both the training and testing datasets have been preprocessed using consistent techniques to ensure data integrity and model reliability. For categorical variables, missing values were handled using a Simple Imputer with the most frequent value, followed by One-Hot Encoding for feature encoding. The target variable was encoded using a Label Encoder.

For numerical features, missing values were imputed using an Iterative Imputer, and the data was standardized using a Robust Scaler to minimize the impact of outliers. Additionally, outliers were detected and removed using the Interquartile Range (IQR) method.

## 4. Building a model to predict patient faitality.

* **Ojbective:** Build baseline models.

Principal Component Analysis (PCA) was applied for dimensionality reduction to capture the most important variance in the dataset while reducing feature complexity. To address class imbalance, the SMOTE-Tomek technique was used, combining oversampling and undersampling methods for better class distribution.

For machine learning models, XGBoost and CatBoost were employed due to their robustness and ability to handle complex, non-linear relationships. Since, the machine learning models dint particualry fit appropritely to the data a deep learning approach was required to further improve the results of the predictive modeling.

In the deep learning pipeline, PyTorch was used to build and train a feed forward neural network for further improved model results.



## 5. Model Optimization & Model Explainability

* **Ojbective:** Optimize the models hyperparameters and perform model expainibilitiy techniques.
  
Optuna was used for hyperparameter optimization to enhance model performance through efficient and automated tuning. To ensure model interpretability, both LIME (Local Interpretable Model-agnostic Explanations) and Integrated Gradients were applied. These techniques provided insights into how different features influenced the model’s predictions, contributing to a better understanding of the model’s decision-making process.

# Requirements
Python 3.x

* Jupyter Notebook
* Pandas
* Matplotlib
* Seaborn
* XGBoost
* Sklearn
* Scipy
* Pytorch
* TorchMetircs
* Optuna
* Lime
* Captum
* Numpy

# Installation
```
pip install pandas matplotlib seaborn xgboost scikit-learn scipy torch torchvision torchaudio torchmetrics optuna lime captum numpy
```

# Conclusion
This project demonstrates the potential of machine learning in predicting patient survival outcomes, including death and liver transplantation. This project is implemented using 2 datasets one for testing and one to training the model. The best-performing model achieved an accuracy of 83% using deep learning and 74% using traditional machine learning, highlighting its promise as a support tool in clinical prognosis and patient management.

# Special Thanks
Miss Annette Nellyet : https://www.linkedin.com/in/annette-nellyet/

> [!TIP]
> Various feature engnnering techniques were applied such as:
> * SOFA score: Calcualtion of SOFA (Sequential Organ Failure Assessment) scores by combining various features.
> * Bins: Categorization of general medical indications and abnormal test results differently for men, women, and children and placing them into appopriate bins.
Unfortunately, non of these feature engneering techniques produced bettter outcomes.
