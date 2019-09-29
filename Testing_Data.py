import os
import numpy as np
import pickle
import time
import json
from frovedis.exrpc.server import FrovedisServer # frovedis
#from sklearn.linear_model import LogisticRegression # sklearn

FrovedisServer.initialize("mpirun -np 4 {}".format(os.environ['FROVEDIS_SERVER'])) # frovedis
i=0
while True:
    try:
        X_train=np.load('/usr/uhome/HT0011/X_Train'+str(i)+'.npy')
    except FileNotFoundError:
        break
    start=time.time()
    clf = pickle.load(open("/usr/uhome/HT0011/model/Model1.sav","rb"))
    y_pred = clf.predict(X_train)
    end=time.time()
    data={"score":y_pred,"time":end-start}
    json_data=json.dumps(data)
    json_data_write=json.dump(data,"/usr/uhome/HT0011/output")
FrovedisServer.shut_down() # frovedis