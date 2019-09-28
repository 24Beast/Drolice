import os
import numpy as np
import pickle
from frovedis.exrpc.server import FrovedisServer # frovedis
from frovedis.mllib.linear_model import LogisticRegression # frovedis
#from sklearn.linear_model import LogisticRegression # sklearn

FrovedisServer.initialize("mpirun -np 4 {}".format(os.environ['FROVEDIS_SERVER'])) # frovedis
X_train=np.load('/usr/uhome/HT0011/X_Train.npy')
y_train=np.load('/usr/uhome/HT0011/y_train.npy')
y_train=y_train[:,0]
y_train = y_train.astype(np.float64)
y_train = 2 * y_train - 1  # frovedis only supports labels of {-1, 1}

C = 10.0
max_iter=10000
solver = "sag"

clf = LogisticRegression(random_state=0, solver=solver, C=C, max_iter=max_iter).fit(X_train, y_train)
y_pred = clf.predict(X_train)
score = 1.0 * sum(y_train == y_pred) / len(y_train)
FrovedisServer.shut_down() # frovedis

pickle.dump(clf, open("/usr/uhome/HT0011/model/Model.sav", 'wb'))

print("score: {}".format(score))