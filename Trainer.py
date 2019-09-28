import os
import numpy as np
import pickle
import time
from frovedis.exrpc.server import FrovedisServer # frovedis
from frovedis.mllib.svm import LinearSVC # frovedis
#from sklearn.linear_model import LogisticRegression # sklearn

FrovedisServer.initialize("mpirun -np 4 {}".format(os.environ['FROVEDIS_SERVER'])) # frovedis
X_train=np.load('/usr/uhome/HT0011/X_Train.npy')
y_train=np.load('/usr/uhome/HT0011/y_train.npy')
y_train=y_train[:,0]
y_train = y_train.astype(np.float64)
y_train = 2 * y_train - 1  # frovedis only supports labels of {-1, 1}
start=time.time()
clf = LinearSVC().fit(X_train, y_train)
y_pred = clf.predict(X_train)
score = 1.0 * sum(y_train == y_pred) / len(y_train)
end=time.time()
FrovedisServer.shut_down() # frovedis

pickle.dump(clf, open("/usr/uhome/HT0011/model/Model1.sav", 'wb'))

print("score: {}".format(score))
print("Model:SVC")
print(end-start)