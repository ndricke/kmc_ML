import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
import sklearn.model_selection as skms
import sklearn.preprocessing as skprep
from sklearn.pipeline import Pipeline



def baseline_model():
    ffnn = Sequential()
    ffnn.add(Dense(36, input_dim=6, activation='relu'))
    ffnn.add(Dense(36, input_dim=36, activation='relu'))
    ffnn.add(Dense(18, input_dim=36, activation='relu'))
    ffnn.add(Dense(1, input_dim=18, activation='linear'))
    ffnn.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    return ffnn

def deeper_model():
    ffnn = Sequential()
    ffnn.add(Dense(36, input_dim=6, activation='relu'))
    ffnn.add(Dense(36, input_dim=36, activation='relu'))
    ffnn.add(Dense(36, input_dim=36, activation='relu'))
    ffnn.add(Dense(18, input_dim=36, activation='relu'))
    ffnn.add(Dense(1, input_dim=18, activation='linear'))
    ffnn.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    return ffnn

def deepest_model_reg():
    ffnn = Sequential()
    ffnn.add(Dense(36, input_dim=6, activation='relu', kernel_regularizer=keras.regularizers.l2(10**-5)))
    ffnn.add(Dense(36, input_dim=36, activation='relu', kernel_regularizer=keras.regularizers.l2(10**-5)))
    ffnn.add(Dense(36, input_dim=36, activation='relu', kernel_regularizer=keras.regularizers.l2(10**-5)))
    ffnn.add(Dense(18, input_dim=36, activation='relu', kernel_regularizer=keras.regularizers.l2(10**-5)))
    ffnn.add(Dense(6, input_dim=18, activation='relu'))
    ffnn.add(Dense(1, input_dim=6, activation='linear'))
    ffnn.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    return ffnn

def deepest_model():
    ffnn = Sequential()
    ffnn.add(Dense(36, input_dim=6, activation='relu'))
    ffnn.add(Dense(36, input_dim=36, activation='relu'))
    ffnn.add(Dropout(0.5))
    ffnn.add(Dense(36, input_dim=36, activation='relu'))
    ffnn.add(Dropout(0.5))
    ffnn.add(Dense(18, input_dim=36, activation='relu'))
    ffnn.add(Dropout(0.5))
    ffnn.add(Dense(6, input_dim=18, activation='relu'))
    ffnn.add(Dense(1, input_dim=6, activation='linear'))
    ffnn.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    return ffnn

def poly_model():
    ffnn = Sequential()
    ffnn.add(Dense(84, input_dim=84, activation='relu', kernel_regularizer=keras.regularizers.l1(10**-5)))
    ffnn.add(Dense(42, input_dim=84, activation='relu', kernel_regularizer=keras.regularizers.l1(10**-5)))
    ffnn.add(Dense(1, input_dim=42, activation='linear'))
    ffnn.compile(loss='mean_squared_error', optimizer='adam', metrics=[])
    return ffnn


def train_test_model(X, Y, model, epoch_num):
    """
    trains tensorflow model on data, and tests using K-cross validation. Prints mean squared error
    Input:
    X (numpy array): feature data
    Y (numpy vector): value to estimate
    model (function): builds and returns tensorflow model
    epoch_num (int): number of epochs to train data on using batch gradient descent
    """
    estimators = []
    estimators.append(('standardize', skprep.StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=model, epochs=epoch_num, batch_size=50, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = skms.KFold(n_splits=3, random_state=rn_seed)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
    print("Results: %.10f (%.10f) MSE" % (results.mean(), results.std()))



if __name__ == "__main__":
    rn_seed = 74
    np.random.seed(rn_seed)


    #ffnn.fit(X_train, Y_train, epochs=1000, batch_size=500)
    #scores = model.evaluate(X_train, Y_train)
    #predictions = model.predict(X_test)

    #infile = sys.argv[1]
    #infile = "/home/nricke/work/tafel/KMC/k_run.csv"
    infile = "/home/nricke/work/tafel/KMC/k_run_prep.csv"
    df = pd.read_csv(infile, index_col="Unnamed: 0")

    X = df[["OA+AO","OB+BO","AA","AB+BA","BB", "B3_Estimate"]].values
    Y = df["OAB+BAO"].values

    # train test split
    X_train, X_test, y_train, y_test = skms.train_test_split(X, Y, test_size=0.2, random_state=1)

    checkpoint_path_d = "training_d/deep-{epoch:04d}.ckpt"
    checkpoint_path_r = "training_r/deep-{epoch:04d}.ckpt"
    d_call = keras.callbacks.ModelCheckpoint(checkpoint_path_d, save_weights_only=True, verbose=1)
    r_call = keras.callbacks.ModelCheckpoint(checkpoint_path_r, save_weights_only=True, verbose=1)

    model_deep = deepest_model()
    model_deep.save_weights(checkpoint_path_d.format(epoch=0))
    deepest_model_history = model_deep.fit(X_train, y_train, epochs=50, batch_size=50, callbacks=[d_call],
                                              validation_data=(X_test, y_test), verbose=2)

    print()
    print("Beginning next model: deepest_model_reg")
    print()

    model_deep_reg = deepest_model_reg()
    model_deep_reg.save_weights(checkpoint_path_r.format(epoch=0))
    deepest_model_history = model_deep_reg.fit(X_train, y_train, epochs=50, batch_size=50, callbacks=[r_call],
                                              validation_data=(X_test, y_test), verbose=2)

    #for model in [baseline_model, deeper_model, deepest_model]:
    #    train_test_model(X, Y, model, 10)

    #train_test_model(X, Y, deepest_model, 200)
    #train_test_model(X, Y, deepest_model, 50)
    #train_test_model(X, Y, deepest_model_reg, 50)

    #poly = skprep.PolynomialFeatures(3)
    #X_poly = poly.fit_transform(X)
    #train_test_model(X_poly, Y, poly_model, 15)
    #train_test_model(X_poly, Y, poly_model, 200)
