import os
import numpy as np
import pickle
import time
import json
from frovedis.exrpc.server import FrovedisServer # frovedis
#from sklearn.linear_model import LogisticRegression # sklearn

i=0
while(i<5):
    FrovedisServer.initialize("mpirun -np 4 {}".format(os.environ['FROVEDIS_SERVER'])) # frovedis
    X_train=np.load('/usr/uhome/HT0011/Drolice/X_Train'+i+'.npy')
    y_train=np.load('/usr/uhome/HT0011/Drolice/y_train'+i+'.npy')
    y_train=y_train[:,0]
    y_train = y_train.astype(np.float64)
    y_train = 2 * y_train - 1  # frovedis only supports labels of {-1, 1}
    start=time.time()
    clf = pickle.load("/usr/uhome/HT0011/model/Model1.sav")
    y_pred = clf.predict(X_train)
    score = 1.0 * sum(y_train == y_pred) / len(y_train)
    end=time.time()
    data={"score":score,"Model":"SVC","time":end-start}
    json_data=json.dumps(data)
    print(json_data)
    FrovedisServer.shut_down() # frovedis
    
    