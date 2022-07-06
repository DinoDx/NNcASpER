import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import Counter
from matplotlib import pyplot

def dataPreprocessing(f, t):

    data = pd.read_csv("data/complete_dataset.csv", sep=",")

    #Data Cleaning
    data = data.dropna()

    # Feature Selection
    todrop_input_data = ["ComponentName", "nameProject", "M_TextualCohesion","M_TextualEntropy","ComplexClass","LargeClass","LazyClass","RefusedBequest","SpaghettiCode"]
    todrop_output_data = ["ComponentName","nameProject","M_CBO","M_CYCLO","M_DIT","M_ELOC","M_FanIn","M_FanIn_1","M_LCOM","M_LOC",
        "M_LOCNAMM","M_NOA","M_NOC","M_NOM","M_NOMNAMM","M_NOPA","M_PMMM","M_PRB","M_WLOCNAMM","M_WMC","M_WMCNAMM","M_TextualCohesion","M_TextualEntropy"]

    x_data = np.array(data.drop(todrop_input_data, axis=1))[f:t]
    y_data = np.array(data.drop(todrop_output_data, axis=1))[f:t]
                
    # Freature Scaling
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x_data = scaler.fit_transform(x_data)

    # Data Balancing (spostato in traning.py)
    '''''
    y_labeled = []

    for target in y_data:
        if(np.array_equal(target, [0,0,0,0,0])):
                y_labeled .append(0)
        elif(np.array_equal(target, [1,0,0,0,0])):
                y_labeled .append(1)
        elif(np.array_equal(target, [0,1,0,0,0])):
                y_labeled .append(2)
        elif(np.array_equal(target, [0,0,1,0,0])):
                y_labeled .append(3)
        elif(np.array_equal(target, [0,0,0,1,0])):
                y_labeled .append(4)
        elif(np.array_equal(target, [0,0,0,0,1])):
                y_labeled .append(5)
    
    le = preprocessing.LabelEncoder()
    y_labeled  = le.fit_transform(y_labeled )


    # Distribution before balancing with SMOTE
    counter = Counter(labeled_y)
    for k,v in counter.items():
        per = v / len(y) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

    pyplot.bar(counter.keys(), counter.values())
    pyplot.show()
    '''''   

    '''''
    # Distribution after balancing with SMOTE    
    x_res, y_res = imblearn.over_sampling.SMOTE().fit_resample(x_data, y_labeled)
    counter = Counter(y_res)
    for k,v in counter.items():
        per = v / len(y_data) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

    pyplot.scatter(counter.keys(), counter.values())
    pyplot.show()
    '''''

    return x_data, y_data
