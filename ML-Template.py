'''
    Simple Template for creating a Neural Network
    Dataset: Kaggle Titanic Set
    Author: Tobias Stiemer
    Date: 11.04.2020
'''

# import os
# Only use this import if its not set in the plaidml.config
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from keras import callbacks
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.models import Sequential
from numpy.random import choice, seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Plot some of the Data
# sns.countplot(x='Pclass', data=df, palette='hls', hue='Survived')
# plt.xticks(rotation=45)
# plt.show()


# ---------------------------------------------------------------------------------- #
# --------------------- Preprocess data to feed it to the model -------------------- #
# ---------------------------------------------------------------------------------- #
from sklearn.utils._testing import set_random_state
from tensorflow_core.python import set_random_seed




def preprocessing():
    # load data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
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
    test = df[pd.isnull(df['Survived'])]
    print(test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.33)
    return x_train, y_train, x_test, y_test, test
preprocessing()

#
# # ---------------------------------------------------------------------------------- #
# # --------------------------- Create Model using Hyperopt -------------------------- #
# # ---------------------------------------------------------------------------------- #
# def create_model(x_train, y_train):
#     seed(42)
#     set_random_seed(42)
#     model = Sequential()
#     # Define input layer
#     model.add(Dense({{choice([4, 8, 12, 16, 32, 64, 128, 256, 512, 1024])}}, input_dim=x_train.shape[1]))
#     model.add(BatchNormalization())
#     model.add(Activation({{choice(['relu', 'sigmoid'])}}))
#     model.add(Dropout({{uniform(0, 1)}}))
#     # Define first hidden layer
#     model.add(Dense({{choice([4, 8, 12, 16, 32, 64, 128, 256, 512, 1024])}}))
#     model.add(BatchNormalization())
#     model.add(Activation({{choice(['relu', 'sigmoid'])}}))
#     model.add(Dropout({{uniform(0, 1)}}))
#
#     # If we choose to add more Layers
#     # if {{choice(['three', 'four'])}} == 'four':
#     #     # define additional hidden layer
#     #     model.add(Dense({{choice([4, 8, 12, 16, 32, 64, 128, 256, 512, 1024])}}))
#     #     model.add(Activation({{choice(['relu', 'sigmoid'])}}))
#     #     model.add(Dropout({{uniform(0, 1)}}))
#     # Define output Layer
#
#     model.add(Dense(1, activation='sigmoid'))
#
#     # Use a dynamic Learning Rate
#     reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss',
#                                             factor=0.2,
#                                             patience=5,
#                                             min_lr=0.001)
#     # Make use of Early Stopping
#     early_stopping = callbacks.EarlyStopping(monitor='val_loss',
#                                              min_delta=0,
#                                              patience=5,
#                                              verbose=0,
#                                              mode='auto',
#                                              baseline=None,
#                                              restore_best_weights=True)
#     model.compile(loss='binary_crossentropy',
#                   metrics=['accuracy'],
#                   optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
#     result = model.fit(x_train, y_train,
#                        batch_size={{choice([32, 64, 128])}},
#                        epochs={{choice([50, 100, 150, 200, 500])}},
#                        verbose=2,
#                        validation_split=0.1,
#                        callbacks=[reduce_lr, early_stopping])
#
#     # get the highest validation accuracy of the training epochs
#     validation_acc = np.amax(result.history['val_acc'])
#     print('Best validation acc of epoch:', validation_acc)
#     model.save("Titanic-Model.model")
#
#     # ---------------------------------------------------------------------------------- #
#     # ------------------------ Plot the evaluation for Accuracy ------------------------ #
#     # ---------------------------------------------------------------------------------- #
#     # plt.plot(result.history['acc'])
#     # plt.plot(result.history['val_acc'])
#     # plt.title('model accuracy')
#     # plt.ylabel('accuracy')
#     # plt.xlabel('epoch')
#     # plt.legend(['train', 'test'], loc='upper left')
#     # plt.show()
#
#     # ---------------------------------------------------------------------------------- #
#     # ------------------------ Plot the evaluation for Loss ------------------------ #
#     # ---------------------------------------------------------------------------------- #
#     # plt.plot(result.history['loss'])
#     # plt.plot(result.history['val_loss'])
#     # plt.title('model loss')
#     # plt.ylabel('loss')
#     # plt.xlabel('epoch')
#     # plt.legend(['train', 'test'], loc='upper left')
#     # plt.show()
#
#     return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}
#
#
# if __name__ == '__main__':
#     best_run, best_model = optim.minimize(model=create_model,
#                                           data=preprocessing,
#                                           algo=tpe.suggest,
#                                           max_evals=30,
#                                           trials=Trials())
#     x_train, y_train, x_test, y_test, test = preprocessing()
#     print("Evalutation of best performing model:")
#     print(best_model.evaluate(x_test, y_test))
#     print("Best performing model chosen hyper-parameters:")
#     print(best_run)