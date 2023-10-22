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

**Purpose of the analysis:** In this scenario, I used Python and deep machine learning techniques within a Jupyter Notebook to develop a neural network model for predicting the success of funding applicants from the nonprofit foundation Alphabet Soup. My goal was to achieve a model accuracy of at least 75% by making changes to the data during the preprocessing stage and in the model compilation.
  
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

As shown above, the "STATUS" column displays "1" 34,794 times and "0" only 5 times, while the "SPECIAL_CONSIDERATIONS" column indicates "N" 34,272 times and "Y" only 27 times. This situation can be problematic, particularly for certain machine learning algorithms. Models may encounter difficulty in learning patterns from the minority class, with the majority class potentially dominating the predictions.
  
  
  - Data Preprocessing (same as original model with changes below):
      - Removed variables: "EIN", "NAME", "STATUS", "SPECIAL_CONSIDERATIONS"

  - Compiling, Training, and Evaluating the Model:

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/3f5e0bc7-6568-4feb-bee3-a8119dec040e)


  - Model Performance:
      - The model accuracy was 74.79% with loss of 52.18%.

-------------------------------------------------

**Model 2: Add More Hidden Layers and Neurons**

In this model, I chose to include additional hidden layers and neurons. This decision is based on the general principle that such an adjustment can enhance the model's accuracy. By adding more layers and neurons, the model's capacity to discern intricate patterns in the data is expanded. This allows the model to better accommodate the training data, capturing complex and non-linear relationships.


  - Data Preprocessing (same as original model)

  - Compiling, Training, and Evaluating the Model:

      ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/89d8d3f1-f061-47ff-a200-82688138e48d)


  - Model Performance:
      - The model accuracy was 74.72% with loss of 51.75%.

-------------------------------------------------

**Model 3: Bin "ASK_AMT" Column and add Epochs**

In this model, I chose to bin the "ASK_AMT" column because I observed a significant skew in the data, with $5,000 occurring 25,398 times in the dataset.:


  ![image](https://github.com/KTamas03/deep-learning-challenge/assets/132874272/02be1826-5bd6-40f7-ab23-aeb9a0ab42be)



Binning can be beneficial when dealing with data that has a skewed or non-uniform distribution. It helps in handling outliers and extreme values by placing them into appropriate bins.

Additionally, I chose to increase the number of epochs. Training a neural network involves adjusting the model's weights to minimize the loss function. Each epoch represents one complete pass through the training dataset. Increasing the number of epochs provides the model with more opportunities to learn from the data. 


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


While Models 1 and 3 showed an improvement over the original model, Model 2 exhibited the opposite effect. The removal of imbalanced columns led to an improvement in the accuracy score as anticipated. Binning the "ASK_AMT" column and increasing the number of epochs were the most effective changes, albeit only slightly. However, adding more hidden layers and neurons resulted in a minor reduction in model accuracy due to overfitting. This can happen when a model becomes overly complex in relation to the size and quality of the training dataset. I didn't reach the accuracy of 75% as hoped, but achieving a top accuracy score of 74.93% for Model 3 is a close and impressive result.

Recommendation for a different model: I recommend considering a logistic regression model to predict an organization's success when receiving funding from Alphabet Soup. The dataset is relatively straightforward and contains a manageable number of categories within each feature. Furthermore, the relationships between the variables are not highly intricate. Logistic regression is a suitable choice because it excels in binary classification problems, making it a practical and interpretable model for this scenario.


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
