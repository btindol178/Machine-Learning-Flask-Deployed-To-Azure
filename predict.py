import numpy as np
import pickle

# make function to use the models from the .sav files 
def predict(pregnancies,glucose,bloodpressure,skinthickness,insulin,bmi,dpfunc,age,model_name):
    filename = model_name + ".sav" # finding local model name 
    model = pickle.load(open(filename,"rb"))# loading model from local storage
    y_pred = model.predict([[
        pregnancies,
        glucose,
        bloodpressure,
        skinthickness,
        insulin,
        bmi,
        dpfunc,
        age
    ]]) # predict 
    return y_pred[0]

    