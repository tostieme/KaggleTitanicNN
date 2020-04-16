'''
    Coding a neural network that will classify survivors of the titanic
    Dataset: Kaggle Titanic Set
    Author: Tobias Stiemer
    Date: 20.03.2020
'''

import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import keras
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from numpy.random import seed
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow_core.python import set_random_seed

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def preprocessing():

    df = pd.concat([train, test], axis=0, sort=True)

    # Encode categorical data
    # convert to category dtype
    df['Sex'] = df['Sex'].astype('category')
    # Convert to category codes
    df['Sex'] = df['Sex'].cat.codes

    # subset all categorical variables which need to be encoded
    categorical = ['Embarked', 'Ticket']
    for var in categorical:
        df = pd.concat([df, pd.get_dummies(df[var], prefix=var)], axis=1)
        del df[var]

    # drop the variable that we wont be using
    df.drop(['Cabin', 'Name', 'PassengerId'], axis=1, inplace=True)

    # Scale Continuous variable
    continuous = ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp']
    scaler = StandardScaler()
    for var in continuous:
        df[var] = df[var].astype('float64')
        df[var] = scaler.fit_transform(df[var].values.reshape(-1, 1))

    # Handle the missing values for age by taking the mean / Could be handled different
    df['Age'].fillna((df['Age'].mean()), inplace=True)

    # Seperate Data into train and test set
    x = df[pd.notnull(df['Survived'])].drop(['Survived'], axis=1)
    y = df[pd.notnull(df['Survived'])]['Survived']
    x_pred = df[pd.isnull(df['Survived'])].drop(['Survived'], axis=1)
    print(x_pred)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.33)
    return x_train, y_train, x_test, y_test, x_pred

# Load Hypertuned Model
model = keras.models.load_model('Titanic-Model.model')
# print(model.summary())

x_train, y_train, x_test, y_test, x_pred = preprocessing()

# evaluate the model
scores = model.evaluate(x_train, y_train)
print(("\nTrain %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100)))


# calculate predictions
def predictions(x_test):
    test['Survived'] = model.predict(x_pred)
    cleared_column = test['Survived'].fillna(0)
    solution = test[['PassengerId', 'Survived']]
    solution['Survived'] = cleared_column
    solution['Survived'] = solution['Survived'].apply(lambda x: round(x, 0)).astype(int)
    # Write predictions into csv
    solution.to_csv("Survivor_Solution.csv", index=False)

predictions(x_test)

