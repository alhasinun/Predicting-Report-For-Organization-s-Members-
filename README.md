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
After all of the values are numerics, then we extract just the values of the column, because we don't need the column head. We extract value of **Result** to variable "y" (Target variable), and then extract the rest to variable "X" (Predictors Variable)

  ```python
     X=train.drop('Result', axis=1).values
     y=train['Result'].values
  ```
The last step in data preparation is to split the data into train and test data. Data splitting is done to separate data when makin the model; train data is for training the model, meanwhile test data is for testing the model to know the model accuracy when facing future unseen data, so we will be sure that the model is the right choice to use when predicting the case.

We are doing it first by importing `train_test_split` library from `scikit learn`
   ```python
      from sklearn.model_selection import train_test_split
   ```

Then, we split each variables (X as predictors, and y as target) into train and test data. So, there will be total of 4 data in the end (X train, X test, y train, and y test). We set the test_size to 0.2 meaning that 20% of the original data will be split into test data, and 80% into train data.
   ```python
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
   ```

## Data Modelling
It's time to do the model prediction. Because there is already a labeled variable, it means that we know the 'target'. By this means, it is a classification problem, because we make a model based on predictors-target variable matching. The model was created using KNN Classifier. The needed argument for KNN is the number of the neighbors `k`. The optimum value for `k` is dependant on the predictors, and target variable. It is unknown to us, and we have to manually try each value to get the optimum value. Luckily, in Python there is a library that can get the optimal value of the parameter, which in this case is k. It is called hyperparameter tuning, and we can use a library called `GridSearchCV` to get the optimum parameter value.
   ```python
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.model_selection import GridSearchCV

      params={'n_neighbors': np.arange(1,20)}
      knn_class = KNeighborsClassifier()
   ```

Here, we instantiate the range where we want to look for the `k` parameter value, which is from 1 to 20. Then we run the GridSearchCV by passing the `knn_class` (classifier), `params` (parameter), and `cv` (cross folds value). And we fit the GridSearchCV to the training data, `X_Train` and `y_train`.
   ```python
      knn_cv = GridSearchCV(knn_class, params, cv=5)
      knn_cv.fit(X_train, y_train)
      print(knn_cv.best_params_)
      print(knn_cv.best_score_)
   ```
By printing best_params_ and best_score_ we can get the most optimum value of the parameters and its accuracy when applied to the model. From the command above, we get the best value for `k` parameter is 18, and its accuracy is 94%.
   <p align="center">
      <img src = "https://user-images.githubusercontent.com/72293844/101736544-2d38e000-3af6-11eb-9858-2beae6e70e25.jpg" />
   </p>
