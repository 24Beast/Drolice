import os
import numpy as np
import pickle
import time
import json
from frovedis.exrpc.server import FrovedisServer # frovedis
from frovedis.mllib.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression # sklearn

FrovedisServer.initialize("mpirun -np 4 {}".format(os.environ['FROVEDIS_SERVER'])) # frovedis
i=0
l=[]
X=np.load('/usr/uhome/HT0011/X_Train.npy')
y=np.load('/usr/uhome/HT0011/y_train.npy')
y=y[:,0]
l=len(X)
while(i<5):
    X_train=np.load('/usr/uhome/HT0011/X_Train'+str(i)+'.npy')
    y_train=np.load('/usr/uhome/HT0011/Drolice/y_train'+str(i)+'.npy')
#    y_train=y_train[:,0]
    y_train = 2 * y_train - 1  # frovedis only supports labels of {-1, 1}
    start=time.time()
    clf = LinearSVC().fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    score = 1.0 * sum(y_train == y_pred) / len(y_train)
    end=time.time()
    data={"score":score,"Model":"SVC","time":end-start}
    json_data=json.dumps(data)
    l.append(data)
print(l)
FrovedisServer.shut_down() # frovedis