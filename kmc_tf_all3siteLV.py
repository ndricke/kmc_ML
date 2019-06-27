import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
import sklearn.model_selection as skms
import sklearn.preprocessing as skprep
from sklearn.pipeline import Pipeline

import kmc_tf
import kmc_ml_util

def skl_mlp_model(al):
    ffnn = Sequential()
    act = "relu"
    #act = "sigmoid"
    ffnn.add(Dense(84, input_dim=6, activation=act, kernel_regularizer=keras.regularizers.l2(al)))
    #ffnn.add(Dense(84, input_dim=84, activation=act, kernel_regularizer=keras.regularizers.l2(al)))
    ffnn.add(Dense(84, input_dim=84, activation=act, kernel_regularizer=keras.regularizers.l2(al)))
    ffnn.add(Dense(42, input_dim=84, activation=act, kernel_regularizer=keras.regularizers.l2(al)))
    ffnn.add(Dense(20, input_dim=42, activation=act, kernel_regularizer=keras.regularizers.l2(al)))
    ffnn.add(Dense(1, input_dim=20, activation='linear'))
    ffnn.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error', 'mean_squared_error'])
    return ffnn


rn_seed = 74
#alpha = 10.**-6
alpha = 0. # regularization didn't seem to improve the model
np.random.seed(rn_seed)

infile = "/home/nricke/work/tafel/KMC/k_run_prep.csv" #the features in this file are not standardized
df = pd.read_csv(infile)

X = df[["OA+AO","OB+BO","AA","AB+BA","BB", "OOA_est", "AOA_est", "AOB_est", "OAB_est", "AAB_est", "BAB_est"]]
B3_list = ["OOA+AOO", "AOA", "AOB+BOA", "OAB+BAO", "AAB+BAA", "BAB"]

for B3_term in B3_list:
    B3_est = B3_term[:3]+"_est" # get 3 letter tag for 3-site term (just first 3 letters)
    X_B3 = X[["OA+AO","OB+BO","AA","AB+BA","BB", B3_est]].values # select out 3-site estimate for this term

    # standardize training data in a way that can be ported to C
    sc_x = skprep.StandardScaler()
    sc_y = skprep.StandardScaler()
    X_std = sc_x.fit_transform(X_B3)
    Y = df[B3_term].values

    # train-test split
    X_train, X_test, y_train, y_test = skms.train_test_split(X_std, Y, test_size=0.2, random_state=1)

    # save checkpoints in case the model has a very low error in the middle rather than end
    checkpoint_path_sk = "training_allB3/%s-{epoch:04d}.ckpt" % B3_term[:3]
    sk_call = keras.callbacks.ModelCheckpoint(checkpoint_path_sk, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
    model_skl = skl_mlp_model(alpha) # create neural network model
    model_skl.save_weights(checkpoint_path_sk.format(epoch=0)) # save model in case it crashes or overfits
    # fit model to data
    skl_model_history = model_skl.fit(X_train, y_train, epochs=150, batch_size=10000, callbacks=[sk_call],
                                              validation_data=(X_test, y_test), verbose=2)

    # save model in form that can be read by tensorflow in C
    model_skl.save("model_skl_%s.h5" % B3_term[:3])

#model_skl.load_weights("training_sk/deep-0120.ckpt")
#scores = model_skl.evaluate(X_test, y_test, verbose=2)
#print(scores)
