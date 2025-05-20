# Cirrhosis-Fatality-Prediction
This repository contains 2 Jupyter Notebooks that demonstrate a Deep Learning as well as a Machine Learning approach to predict patient survival outcomes based on clinical data. The primary objective is to classify patients into one of three categories:

0 = D (Death)

1 = C (Censored)

2 = CL (Censored due to Liver Transplantation)

# Introduction
Accurately predicting patient outcomes is crucial for clinical decision-making and resource planning. In this project, we utilize patient data to build a predictive model that determines the survival status based on various clinical features.

The dataset includes medical information, presenting correlations between different variables and patient outcomes. By predicting whether a patient will survive, die, or undergo liver transplantation, healthcare providers can make more informed and timely interventions.

# Requirements
Python 3.x

Jupyter Notebook

Pandas

Matplotlib

Seaborn

XGBoost

Sklearn

Scipy

Pytorch

TorchMetircs

Optuna

Lime

Captum

Numpy

# Installation
```
pip install pandas matplotlib seaborn xgboost scikit-learn scipy torch torchvision torchaudio torchmetrics optuna lime captum numpy
```

# Conclusion
This project demonstrates the potential of machine learning in predicting patient survival outcomes, including death and liver transplantation. This project is implemented using 2 datasets one for testing and one to training the model. The best-performing model achieved an accuracy of 83% using deep learning and 74% using traditional machine learning, highlighting its promise as a support tool in clinical prognosis and patient management.
