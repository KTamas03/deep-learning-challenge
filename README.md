## deep-learning-challenge
**Module 21 Challenge - deep-learning**

**Repository Folders and Contents:**
- checkpoints:
  - weights.##.hdf5 files
- checkpoints_model1:
  - weights.##.hdf5 files
- checkpoints_model2:
  - weights.##.hdf5 files
- checkpoints_model3:
  - weights.##.hdf5 files
- AlphabetSoupCharity.h5
- AlphabetSoupCharity_Optimisation.h5
- AlphabetSoupCharity_Optimisation.ipynb
- AlphabetSoupCharity_Original.ipynb

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

![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/95580398-0aa0-4ca9-8327-f7b3fc7e2155)


The target variable I wanted to predict was "IS_SUCCESSFUL", which is binary and indicates whether an organisation is unsuccessful (0) or successful (1).
  
**Resource File I Used:**
  - charity_data.csv

**My Jupyter Notebook Python Scripts:**
  - AlphabetSoupCharity_Original.ipynb (Original Model)
  - AlphabetSoupCharity_Optimisation.ipynb (Models 1, 2 and 3)

**Tools/Libraries I Imported:**
  - import pandas as pd # To read and manipulate the lending data as a dataframe
  - from sklearn.model_selection import train_test_split # To split the dataset into training and testing data
  - from sklearn.preprocessing import StandardScaler # To scale dataset
  - import tensorflow as tf # To build and work with deep learning machine models
  - from tensorflow.keras.callbacks import ModelCheckpoint # Import the ModelCheckpoint callback from tensorflow.keras.callbacks

## Results

**Original Model:**

  - Data Preprocessing:
      - Target variable: "IS_SUCCESSFUL"
      - Feature variables: "APPLICATION_TYPE", "AFFILIATION", "CLASSIFICATION",	"USE_CASE",	"ORGANIZATION",	"STATUS, "INCOME_AMT", "SPECIAL_CONSIDERATIONS", "ASK_AMT"
      - Removed variables: "EIN", "NAME"
      - Binned variables: "APPLICATION_TYPE" - 9 bins, "CLASSIFICATION" - 6 bins
      - The categorical columns were converted to numeric
      - The data was split into features and target arrays
      - The data was also then split into training and test datasets
      - The data was then scaled

  - Compiling, Training, and Evaluating the Model:

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/a9f41252-56c5-483c-a96f-e2bfe39fb767)


  - Model Performance:
      - The model accuracy was 74.76% with loss of 52.11%.

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/0d64abd3-b6c7-4eb2-8867-9a23c4574ad5)

*Note. The original model was used as a benchmark for the next 3 models.*

**Model 1: Remove "SPECIAL_CONSIDERATIONS" column**
 
  - Data Preprocessing (same as original model with changes below):
      - Removed variables: "EIN", "NAME", "SPECIAL_CONSIDERATIONS"

  - Compiling, Training, and Evaluating the Model:

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/7d21f741-1b00-4cef-a1fa-9d30ab04f448)


  - Model Performance:
      - The model accuracy was 74.78% with loss of 52.08%.
      
      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/b6657a92-c316-4306-a4dd-ff4fb30e1ffb)


**Model 2: Add More Hidden Layers and Neurons**

  - Data Preprocessing (same as original model)

  - Compiling, Training, and Evaluating the Model:

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/c3d4bff5-e2e2-4a56-8f0a-a07f90bc0e81)


  - Model Performance:
      - The model accuracy was 74.82% with loss of 51.79%.

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/b11ec779-6553-46fc-a599-4c840ef90876)


**Model 3: Bin "ASK_AMT" Column and add EPOCs**

  - Data Preprocessing (same as original model with changes below):
      - Binned variables: "APPLICATION_TYPE" - 9 bins, "CLASSIFICATION" - 6 bins, "ASK_AMT" - 3 bins

  - Compiling, Training, and Evaluating the Model:

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/a2819266-e412-495e-abca-0c0faa01f362)


  - Model Performance:
      - The model accuracy was 74.94% with loss of 51.47%.
      
      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/5932e6cf-a6d3-4700-adeb-c1d4258f948a)

  
## Summary

Overall, model??? performed the best because...
Summary: Summarise the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/0d64abd3-b6c7-4eb2-8867-9a23c4574ad5)

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/9e8dc55c-3ee0-4827-8c4c-e19e1e8193f2)

| Score | Original Model | Model 1 | Model 2 | Model 3 |
|:--------------:|:--------------: |:--------------:|:--------------:|:--------------:|
| Accuracy | 74.76% | 74.78% | 74.82% | 74.94% |
| Loss | 52.11% | 52.08% | 51.79% | 51.47% |


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
