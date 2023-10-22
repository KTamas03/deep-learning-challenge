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

**Purpose of the analysis:** In this scenario, I employed python and deep machine learning techniques within a Jupyter Notebook to develop a neural network model to predict whether applicants' of funding will be successful if they receive funding from nonprofit foundation Alphabet Soup. I aimed to achieve a model of 75% accuracy whilst making changes to the data in preprocessing stage and in the model compilation.
  
**Data Used:** I worked with a charity dataset (https://static.bc-edx.com/data/dla-1-2/m21/lms/starter/charity_data.csv) with over 34,000 organisations that had received funds from Alphabet Soup. This dataset contained the following fields:

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

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/4ff869cf-7142-4f9b-b8e2-5875337063df)

        
  - Model Performance:
      - The model accuracy was 74.76% with loss of 52.11%.


*Note. The original model was used as a benchmark for the next 3 models.*

-------------------------------------------------

**Model 1: Remove "STATUS" and "SPECIAL_CONSIDERATIONS" columns**

 In this model, I decided to remove the "STATUS" and "SPECIAL CONSIDERATION" columns as there appeared to be an extreme imbalance in the data:
 
 ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/560d8180-9711-4da6-a8eb-97812ba5113e)

 As shown above, the "STATUS" column shows "1" 34,794 times, and "0" only 5 times, whilst the "SPECIAL_CONSIDERATIONS" column shows "N" 34,272 times, and "Y" only 27 times. This can be problematic, especially for some machine learning algorithms. Models may struggle to learn patterns from the minority class, and the majority class can dominate the predictions.
  
  
  - Data Preprocessing (same as original model with changes below):
      - Removed variables: "EIN", "NAME", "STATUS", "SPECIAL_CONSIDERATIONS"

  - Compiling, Training, and Evaluating the Model:

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/3f5e0bc7-6568-4feb-bee3-a8119dec040e)


  - Model Performance:
      - The model accuracy was 74.79% with loss of 52.18%.

-------------------------------------------------

**Model 2: Add More Hidden Layers and Neurons**

In this model, I decided to add more hidden layers and neurons as generally this can improve the accuracy of the model. Adding more layers and neurons increases the model's capacity to learn complex patterns in the data. The model becomes more capable of fitting the training data, including intricate and non-linear relationships.


  - Data Preprocessing (same as original model)

  - Compiling, Training, and Evaluating the Model:

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/89d8d3f1-f061-47ff-a200-82688138e48d)


  - Model Performance:
      - The model accuracy was 74.72% with loss of 51.75%.

-------------------------------------------------

**Model 3: Bin "ASK_AMT" Column and add Epochs**

In this model, I decided to bin the "ASK_AMT" column as I could see there data was heavily skewed, with $5,000 occurring 25,398 times in the dataset:


  ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/02be1826-5bd6-40f7-ab23-aeb9a0ab42be)



Binning can be beneficial when dealing with data that has a skewed or non-uniform distribution. It helps in dealing with outliers and extreme values by placing them into appropriate bins.

In addition to this, I decided to also increase the number of Epochs. Training a neural network is like learning from examples. Each epoch is like going through all the examples once. When you increase the number of epochs, it's like practicing more, giving the model more chances to get better by learning from the examples.


  - Data Preprocessing (same as original model with changes below):
      - Binned variables: "APPLICATION_TYPE" - 9 bins, "CLASSIFICATION" - 6 bins, "ASK_AMT" - 3 bins


  - Compiling, Training, and Evaluating the Model:

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/d669c176-846d-4fa8-982b-a85bae0c8232)

  
- Model Performance:
      - The model accuracy was 74.93% with loss of 51.50%.
      

## Summary

Overall, Model 3 performed the best with the highest accuracy score and lowest loss as shown in the table below:


| Score | Original Model | Model 1 | Model 2 | Model 3 |
|:--------------:|:--------------: |:--------------:|:--------------:|:--------------:|
| Accuracy | 74.76% | 74.79% | 74.72% | 74.93% |
| Loss | 52.11% | 52.18% | 51.75% | 51.50% |


Whilst Model 1 and 3 saw an improvement on the Original model, Model 2 was the opposite.
Removing the imbalance columns improved the accuracy score as hoped. Binning the ASK_AMT column and increasing the Epochs was the most effective, but only slightly.
Adding more hidden layers and neurons resulted in a minor reduction of the accuracy of the model due to overfitting. This occurs when a model becomes too complex relative to the size and quality of the training dataset.

Recommendation for a different model: logistic regression model could be suitable to predict an organisation's success if they receive funding from Alphabet Soup. The dataset is fairly simple and doesn't have too many categories within each feature. ALso the relationships between the variables are not too complex.


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
  - How to define a callback to sace the model's weights every five epochs: https://www.tensorflow.org/tutorials/keras/save_and_load
