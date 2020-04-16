#coding=utf-8

try:
    import numpy as np
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras import callbacks
except:
    pass

try:
    from keras.layers import Dense, Activation, Dropout, BatchNormalization
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from numpy.random import choice, seed
except:
    pass

try:
    from sklearn.model_selection import train_test_split
except:
    pass

try:
    from sklearn.preprocessing import StandardScaler
except:
    pass

try:
    from sklearn.utils._testing import set_random_state
except:
    pass

try:
    from tensorflow_core.python import set_random_seed
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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


def keras_fmin_fnct(space):

    seed(42)
    set_random_seed(42)
    model = Sequential()
    # Define input layer
    model.add(Dense(space['Dense'], input_dim=x_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation(space['Activation']))
    model.add(Dropout(space['Dropout']))
    # Define first hidden layer
    model.add(Dense(space['Dense_1']))
    model.add(BatchNormalization())
    model.add(Activation(space['Activation_1']))
    model.add(Dropout(space['Dropout_1']))

    # If we choose to add more Layers
    # if space['Dropout_2'] == 'four':
    #     # define additional hidden layer
    #     model.add(Dense(space['Dense_2']))
    #     model.add(Activation(space['Activation_2']))
    #     model.add(Dropout(space['Dropout_3']))
    # Define output Layer

    model.add(Dense(1, activation='sigmoid'))

    # Use a dynamic Learning Rate
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.2,
                                            patience=5,
                                            min_lr=0.001)
    # Make use of Early Stopping
    early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                                             min_delta=0,
                                             patience=5,
                                             verbose=0,
                                             mode='auto',
                                             baseline=None,
                                             restore_best_weights=True)
    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=space['optimizer'])
    result = model.fit(x_train, y_train,
                       batch_size=space['batch_size'],
                       epochs=space['epochs'],
                       verbose=2,
                       validation_split=0.1,
                       callbacks=[reduce_lr, early_stopping])

    # get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    model.save("Titanic-Model.model")

    # ---------------------------------------------------------------------------------- #
    # ------------------------ Plot the evaluation for Accuracy ------------------------ #
    # ---------------------------------------------------------------------------------- #
    # plt.plot(result.history['acc'])
    # plt.plot(result.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    # ---------------------------------------------------------------------------------- #
    # ------------------------ Plot the evaluation for Loss ------------------------ #
    # ---------------------------------------------------------------------------------- #
    # plt.plot(result.history['loss'])
    # plt.plot(result.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
        'Dense': hp.choice('Dense', [4, 8, 12, 16, 32, 64, 128, 256, 512, 1024]),
        'Activation': hp.choice('Activation', ['relu', 'sigmoid']),
        'Dropout': hp.uniform('Dropout', 0, 1),
        'Dense_1': hp.choice('Dense_1', [4, 8, 12, 16, 32, 64, 128, 256, 512, 1024]),
        'Activation_1': hp.choice('Activation_1', ['relu', 'sigmoid']),
        'Dropout_1': hp.uniform('Dropout_1', 0, 1),
        'Dropout_2': hp.choice('Dropout_2', ['three', 'four']),
        'Dense_2': hp.choice('Dense_2', [4, 8, 12, 16, 32, 64, 128, 256, 512, 1024]),
        'Activation_2': hp.choice('Activation_2', ['relu', 'sigmoid']),
        'Dropout_3': hp.uniform('Dropout_3', 0, 1),
        'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'epochs': hp.choice('epochs', [50, 100, 150, 200, 500]),
    }
