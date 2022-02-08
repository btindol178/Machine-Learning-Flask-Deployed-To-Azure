import numpy as np
import pickle

# Need this specific version because of package ._base problems 
#  pip install scikit-learn==0.22.1


# just for now jsut this model
filename = "KNN.sav"
model = pickle.load(open(filename,"rb"))

# pick column out of preprocessing file 0 outcome and this array is 1 outcome [3,78,50,32,88,31,0.248,26]
y_pred = model.predict(np.array([[1,89,66,23,94,28.1,0.167,21]]))

print("KNN","\tOutcome:",y_pred[0])

