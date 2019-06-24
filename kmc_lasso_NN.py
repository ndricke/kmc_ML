import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import rcParams


import sklearn.preprocessing as skprep
import sklearn.linear_model as sklm
import sklearn.model_selection as skms
import sklearn.metrics as skm
from sklearn.decomposition import PCA
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor



font = {'size':18}
mpl.rc('font',**font)
rcParams['figure.figsize'] = 10,10


def B2_convert(site1, site2):
    assert type(site1) == str and type(site2) == str
    assert len(site1) == 1 and len(site2) == 1
    B3_dict = {'O':1, 'A':2, 'B':3}
    if site1 == site2:
        return site1*2
    else:
        site1_order = B3_dict[site1]; site2_order = B3_dict[site2];
        if site2_order < site1_order:
            site1, site2 = site2, site1
        return site1+site2+"+"+site2+site1

def B3_convert(B3):
    # B3 is a 3 letter string representing the 3-site correlation term
    assert type(B3) == str and len(B3) == 3
    B3_dict = {'O':1, 'A':2, 'B':3}
    if B3[0] == B3[2]:
        return B3
    else:
        return B3 + "+" + B3[::-1]

"""
label_string = "t,A,B,OO,OA+AO,OB+BO,AA,AB+BA,BB,OOO,OOA+AOO,OOB+BOO,AOA,AOB+BOA,BOB,OAO,OAA+AAO,OAB+BAO,AAA,AAB+BAA,BAB,OBO,OBA+ABO,OBB+BBO,ABA,ABB+BBA,BBB"
labels = label_string.split(',')
print(labels)

#indir = "/home/nricke/work/tafel/KMC/500x_rep/"
indir = "/home/nricke/work/tafel/KMC/LV/k_run"

df_list = []
#for infile in os.listdir(indir):
for path, subdirs, files in os.walk(indir):
    print(path, subdirs, files)
    for name in files:
        if name[:3] == "run" and name[-3:] == "csv":
            df_list.append(pd.read_csv(path+'/'+name, header=None, names=labels))

print(len(df_list))

df = pd.concat(df_list)
df.to_csv("k_run.csv")
sys.exit(-1)
"""

ep = 10.**-10

df = pd.read_csv("~/work/tafel/KMC/k_run.csv")
B3_string = sys.argv[1] # the 3-site correlation term to estimate

assert len(B3_string) == 3 #this model is only made to predict 3-site terms

# construct terms for B3_Estimate function that Andrew has been using
# map A,A to [AA], A,B and B,A to [AB+BA]
# if left term i same as middle term j, get ii
# if left term i different from middle term j, get ij+ji or ji+ij (terms always ordered O,A,B?)

df['O'] = 1. - df.A - df.B
B3_middle = B3_string[1]

B3_left2 = B2_convert(B3_string[0], B3_string[1])
B3_right2 = B2_convert(B3_string[1], B3_string[2])
B3 = B3_convert(B3_string)

#df_x = df[["A","B","OO","OA+AO","OB+BO","AA","AB+BA","BB"]].values

#df_x = df[["A","B","OA+AO","OB+BO","AA","AB+BA","BB"]]
#df_x["A"] = 1./(df_x["A"]+ep)
#df_x["B"] = 1./(df_x["B"]+ep)
#df_x["O"] = 1./(df_x["O"]+ep)

df["OA+AO"] = df["OA+AO"]/2.
df["OB+BO"] = df["OB+BO"]/2.
df["AB+BA"] = df["AB+BA"]/2.

df["B3_Estimate"] = df[B3_left2]*df[B3_right2]/(df[B3_middle])
df["B3sub_Estimate"] = df[B3_left2]*df[B3_string[2]] + df[B3_right2]*df[B3_string[0]] - df[B3_string[0]]*df[B3_string[1]]*df[B3_string[2]]

#df.to_csv("k_run_prep.csv")

#df_x = df[["OA+AO","OB+BO","AA","AB+BA","BB", "B3_Estimate"]].values
#df_x = df[["OA+AO","OB+BO","AA","AB+BA","BB"]].values
#df_x = df[["B3_Estimate"]].values
df_x = df[["B3sub_Estimate"]].values

#df = df[["OA+AO","OB+BO","AA","AB+BA","BB","OAB+BAO","AAB+BAA","BAB","OBA+ABO","ABA","ABB+BBA","BBB","AAA"]]

"""
"OAB+BAO" "AAB+BAA" "BAB" "OBA+ABO" "ABA" "ABB+BBA" "BBB" "AAA"
"""

df_B3 = df[B3]

#poly = skprep.PolynomialFeatures(3)
#X_poly = poly.fit_transform(df_x)
X_poly = df_x

y = np.array(df_B3)

sc_x = skprep.StandardScaler()
sc_y = skprep.StandardScaler()
X_std = sc_x.fit_transform(X_poly)
#print(sc_x.mean_)
#print(sc_x.scale_)
#print(X_std.mean(), X_std.std())
#X_std = X_poly

X_train, X_test, y_train, y_test = skms.train_test_split(X_std, y, test_size=0.2, random_state=1)
#print(X_std)
#y_std = sc_y.fit_transform(y)

print("Feature shape: ", X_train.shape)
print("y: ", y)

#pca = PCA(0.9999).fit(X_std)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel("number of components")
#plt.ylabel("cumulative explained variance")
#plt.show()

#X_std = pca.transform(X_std)
#print(X_std.shape)

al = 10.**-5
#alpha_list = [10.**-4, 5*10.**-5, 10.**-5]
#for al in alpha_list:
#regr = sklm.ElasticNet(alpha=al, l1_ratio=0.9, random_state=0)
#regr = sklm.Lasso(alpha=al, random_state=0, max_iter=4000)
regr = sklm.LinearRegression()
#regr = KernelRidge(alpha=10.**-5, kernel="RBF")

#regr = MLPRegressor(hidden_layer_sizes=(50,20,8), random_state=1, alpha=al)


layers = [(84,42)]
#regr = MLPRegressor(hidden_layer_sizes=(84,42,20), random_state=1, alpha=al, max_iter=20000)
#layers = [(84,42), (84,84), (168,84)]
#layers = [(84,42,8), (40,20,10), (20,20,20)]
#layers = [(168,168), (40,40,40)]
#for layer in layers:
#    regr = MLPRegressor(hidden_layer_sizes=layer, alpha=al, max_iter=4000)
#
#    kf = skms.KFold(n_splits=5)
#    results = skms.cross_val_score(regr, X_std, y, cv=kf, scoring="neg_mean_squared_error")
#    print(layer, np.mean(results), np.std(results))

regr.fit(X_train, y_train)
y_fit = regr.predict(X_train)
y_pred = regr.predict(X_test)
#print('Coefficients: \n', regr.coef_)
print("Mean squared error train: %.8f" % skm.mean_squared_error(y_train, y_fit))
print("Mean squared error test: %.8f" % skm.mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % skm.r2_score(y_test, y_pred))

#print(poly.powers_)
#ppow = poly.powers_
#for i in range(len(regr.coef_)):
#    if regr.coef_[i] > ep:
#        print(regr.coef_[i], ppow[i,:], sc_x.mean_[i], sc_x.scale_[i])


plt.scatter(y_fit, y_train, color='blue', label="Training Data")
plt.scatter(y_pred, y_test, color='orange', label="Testing Data")

plt.plot([-1,1],[-1,1], color="black")
#plt.ylabel("%s from KMC" % B3)
plt.ylabel("Kinetic Monte Carlo")
plt.xlabel("Fit")
plt.xlim([-0.004,np.max(y_pred)+0.005])
plt.ylim([-0.004, np.max(y)+0.005])
plt.legend()

#plt.show()
plt.savefig("OAB_B3subest_LL_krun.png", transparent=True, bbox_inches='tight', pad_inches=0.05)



#"""
