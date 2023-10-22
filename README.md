## deep-learning-challenge
**Module 21 Challenge - deep-learning**

**Repository Folders and Contents:**
- Credit_Risk:
  - Resources:
    - lending_data.csv
  - credit_risk_classification.ipynb


## Table of Contents

- [Overview of the Analysis](#overview-of-the-analysis)
- [Results](#results)
- [Summary](#summary)
- [Getting Started](#getting-started)
- [Installing](#installing)
- [Contributing](#contributing)


## Overview of the Analysis

**Purpose of the analysis:** In this scenario, I employed python and deep machine learning techniques within a Jupyter Notebook to develop a neural network model to predict whether applicants' of funding will be successful if they receive funding from nonprofit foundation Alphabet Soup. I aimed to assess whether an organisation should be categorized as a "successful" or "unsuccessful" based on the available data.
  
**Data Used:** I worked with a charity dataset (https://static.bc-edx.com/data/dla-1-2/m21/lms/starter/charity_data.csv) containing over 34,000 organisations that had received funds from Alphabet Soup. This dataset contained the followined fields:
![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/7b15f50b-727c-4e9a-9c70-5522b1f5c6e7)

The target variable I wanted to predict was "IS_SUCCESSFUL", which is binary and indicates whether an organisation is unsuccessful (0) or successful (1).
  
**Resource File I Used:**
  - charity_data.csv

**My Jupyter Notebook Python Scripts:**
  - AlphabetSoupCharity_Original.ipynb
  - AlphabetSoupCharity_Optimisation.ipynb

**Tools/Libraries I Imported:**
  - import pandas as pd # To read and manipulate the lending data as a dataframe
  - from sklearn.model_selection import train_test_split # To split the dataset into training and testing data
  - from sklearn.preprocessing import StandardScaler # To scale dataset
  - import tensorflow as tf # To build and work with deep learning machine models
  - from tensorflow.keras.callbacks import ModelCheckpoint # Import the ModelCheckpoint callback from tensorflow.keras.callbacks

## Results

**Original Model:**

*Data Preprocessing:*
  - Target variable: "IS_SUCCESSFUL"
  - Feature variables: "APPLICATION_TYPE", "AFFILIATION", "CLASSIFICATION",	"USE_CASE",	"ORGANIZATION",	"STATUS, "INCOME_AMT", "SPECIAL_CONSIDERATIONS", "ASK_AMT"
  - Removed variables: "EIN", "NAME"
  - Binned variables: "APPLICATION_TYPE" - 9 bins, "CLASSIFICATION" - 6 bins
  - The categorical columns were converted to numeric
  - The data was split into features and target arrays
  - The data was also then split into training and test datasets
  - The data was then scaled

*Compiling, Training, and Evaluating the Model:*

I chose to have two hidden layers using "relu" activation function for both with 80 neurons for the first one and 30 neurons for the second. I also had an outer layer using sigmoid activation function with 1 neuron.
![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/cca16c91-48ed-4668-a296-6fdef9d66b44)

Model Performance:
![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/5f7e06a5-5205-4371-9460-ce784f170233)

The original model was used as a benchmark for the next 3 models.

**Model 1:**
The matrix below once again reveals a very high correlation between the independent variables, as indicated by the predominantly high Pearson correlation coefficient values, most of which are over 0.80. This suggests the presence of multicollinearity within the lending_df dataframe.


**Model 2:**
The extremely high VIF values (any score exceeding 5) indicate the presence of multicollinearity within the lending_df dataframe. This implies that accurately determining the coefficients for each independent variable and their true impact on the dependent variable will be challenging. Additionally, there is potential for overfitting, meaning the model may capture noise in the data due to highly correlated variables, rather than the genuine underlying relationships.


**Model 3:**
The confusion matrix shows that the model correctly predicted the vast majority of healthy loans in the dataset (18663). The model also accurately predicted 563 high-risk loans. However, there were 102 false positives, meaning the model incorrectly predicted high-risk loans that were actually healthy loans. Additionally, there were 56 false negatives, indicating cases where the model incorrectly predicted healthy loans that were actually high-risk loans.


## Summary

Overall, model??? performed the best because...
Summary: Summarise the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

## Getting Started

**Programs/software I used:**
  - Jupyter Notebook: python programming tool, was used for data manipulation and consolidation.

**To activate dev environment and open Jupyter Notebook:**
  - Open Anaconda Prompt
  - Activate dev environment, type 'conda activate dev'
  - Navigate to the folder where repository is saved on local drive
  - Open Jupyter Notebook, type 'Jupyter Notebook'

## Installing

**Install scikit-learn library**
  - https://scikit-learn.org/stable/install.html
  
## Contributing
  - How to create pairplots plots: https://seaborn.pydata.org/generated/seaborn.pairplot.html
  - How to create correlation matrix heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html
  - How to calculate variation inflation factor: https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
