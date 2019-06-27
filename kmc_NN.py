
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import rcParams

import sklearn.preprocessing as skprep
import sklearn.model_selection as skms
import sklearn.metrics as skm
from sklearn.neural_network import MLPRegressor

import kmc_ml_util

font = {'size':18}
mpl.rc('font',**font)
rcParams['figure.figsize'] = 10,10


#al = 10.**-6 # L2 regularization for neural network
al = 0.
B3_string = sys.argv[1] # the 3-site correlation term to estimate

df = pd.read_csv("~/work/tafel/KMC/k_run.csv")
assert len(B3_string) == 3 #this model is only made to predict 3-site terms

# construct terms for B3_Estimate function that Andrew has been using
df['O'] = 1. - df.A - df.B
B3_middle = B3_string[1]
B3_left2 = kmc_ml_util.B2_convert(B3_string[0], B3_string[1])
B3_right2 = kmc_ml_util.B2_convert(B3_string[1], B3_string[2])
B3 = kmc_ml_util.B3_convert(B3_string)

#ep = 10.**-10
#df_x["A"] = 1./(df_x["A"]+ep)
#df_x["B"] = 1./(df_x["B"]+ep)
#df_x["O"] = 1./(df_x["O"]+ep)

df["OA+AO"] = df["OA+AO"]/2.
df["OB+BO"] = df["OB+BO"]/2.
df["AB+BA"] = df["AB+BA"]/2.

df["B3_Estimate"] = df[B3_left2]*df[B3_right2]/(df[B3_middle])
df["B3sub_Estimate"] = df[B3_left2]*df[B3_string[2]] + df[B3_right2]*df[B3_string[0]] - df[B3_string[0]]*df[B3_string[1]]*df[B3_string[2]]

df_x = df[["OA+AO","OB+BO","AA","AB+BA","BB", "B3_Estimate"]].values
df_B3 = df[B3]

y = np.array(df_B3)

sc_x = skprep.StandardScaler() # standardize input data
X_std = sc_x.fit_transform(df_x)
#sc_y = skprep.StandardScaler()
#y = sc_y.fit_transform(y.reshape(-1,1)).flatten() # scaling target values shouldn't generally be necessary

X_train, X_test, y_train, y_test = skms.train_test_split(X_std, y, test_size=0.2, random_state=1)

print("Feature shape: ", X_train.shape)
print("y: ", y)

regr = MLPRegressor(hidden_layer_sizes=(84,42,20), random_state=2, alpha=al, max_iter=200)

regr.fit(X_train, y_train)
print(regr.n_iter_) # number of training epochs before fit terminated
y_fit = regr.predict(X_train)
y_pred = regr.predict(X_test)
print("Mean squared error train: %.8f" % skm.mean_squared_error(y_train, y_fit))
print("Mean squared error test: %.8f" % skm.mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % skm.r2_score(y_test, y_pred))

plt.scatter(y_fit, y_train, color='blue', label="Training Data")
plt.scatter(y_pred, y_test, color='orange', label="Testing Data")

plt.plot([-1,1],[-1,1], color="black")
#plt.ylabel("%s from KMC" % B3)
plt.ylabel("Kinetic Monte Carlo")
plt.xlabel("Fit")
plt.xlim([-0.004,np.max(y_pred)+0.005])
plt.ylim([-0.004, np.max(y)+0.005])
plt.legend()

plt.show()
#plt.savefig("OAB_B3subest_LL_krun.png", transparent=True, bbox_inches='tight', pad_inches=0.05)
