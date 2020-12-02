# Predicting Report For Organization's Members

This is my second code in github. Github is my place to exercise my skills in coding, so i will keep posting in here to implement my knowledge in programming related to data analysis or data science.

In this repository, i still use Python just like before. But, unlike before, this time i am doing data science problem. I am implementing machine learning method to predict progress report of organization's members based on the parameters of the previous report. For this case, i am using classification of supervised learning, because i try to predict the report based on the previous report which included previous parameters and labels.

Here are the steps to do that.

### Initialization
I am using Python with Jupyter Notebook, so not all of the libraries are initialized in the beginning. This is just to initialize basic libraries (`pandas` and `numpy`)

   ```python
      import pandas as pd
      import numpy as np
   ```

## Data Preparation

  ```python
     train_raw=pd.read_csv(r'D:\Program_Files\python\train_report.csv')
  ```
We are using reports data from previous batch (2015-2017) to build a classification model. The data contains 9 columns, with the "Result" column describes the progress result of each member (very bad, bad, average, good, very good). Here is what the data looks like (The "Name" column is censored because it contains real name of a person).

   <p align="center">
      <img src = "https://user-images.githubusercontent.com/72293844/100861441-6785fa00-34c4-11eb-8cf3-e568b14486de.jpg" />
   </p>!

"NIM" and "Name" columns can be discarded because they are not needed in the process. The only needed parameters for the model are **Participation**, **Professional Development**, **Discipline**, **Contribution**, **Critical Thinking**, and **Score**. **Result** will be used as the target label to fit and train the model.

  ```python
     train = train_raw.drop(['Name', 'NIM'], axis=1)
     print(train.head())
  ```
As it can be seen, here is the needed dataframe
   <p align="center">
      <img src = "https://user-images.githubusercontent.com/72293844/100863088-9e5d0f80-34c6-11eb-96f5-32c5783f404d.jpg" />
   </p>

We want the **Result** column to be numerics, because the model will not be train if it is not numerics, so we change the string to numeric (integer).

  ```python
     clean_result = {'Result': {'Very Bad':0, 'Bad':1, 'Average':2, 'Good':3, 'Very Good':4}}
     train.replace(clean_result, inplace=True)
  ```
After all of the values are numerics, then we extract just the values of the column, because we don't need the column head. We extract value of **Result** to variable "y" (Target variable), and then extract the rest to variable "X" (Trained Variable)

  ```python
     X=train.drop('Result', axis=1).values
     y=train['Result'].values
  ```
  
