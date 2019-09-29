import os
import numpy as np
import time
#import json
from frovedis.exrpc.server import FrovedisServer # frovedis
from frovedis.mllib.svm import LinearSVC 
#from sklearn.linear_model import LogisticRegression # sklearn

FrovedisServer.initialize("mpirun -np 4 {}".format(os.environ['FROVEDIS_SERVER'])) # frovedis
X_train=np.load('/usr/uhome/HT0011/X_Train.npy')
y_train=np.load('/usr/uhome/HT0011/y_train.npy')
y_train=y_train[:,0]
y_train = y_train.astype(np.float64)
y_train = 2 * y_train - 1  # frovedis only supports labels of {-1, 1}
start=time.time()
clf = LinearSVC().fit(X_train, y_train)
end=time.time()
y_pred = clf.predict(X_train)
score = 1.0 * sum(y_train == y_pred) / len(y_train)

X_train=np.load('/usr/uhome/HT0011/X_Train0.npy')
start1=time.time()
y_pred1 = clf.predict(X_train)
end1=time.time()
data={"score":y_pred}

#json_data=json.dumps(data)
#json_data_write=json.dump(data,"/usr/uhome/HT0011/output")
 
print("Model:SVC")
print("Training Data Acc:"+str(score))
print("time taken to train:"+str(end-start)+"secs")
print("time taken to predict"+str(end1-start1)+"secs")
FrovedisServer.shut_down() # frovedis