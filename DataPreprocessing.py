import pandas as pd
import numpy as np
from sklearn import preprocessing

def dataPreprocessing(n):

    data = pd.read_csv("complete_Dataset.csv", sep=",")
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    useless_input_data = ["ComponentName", "nameProject", "M_TextualCohesion","M_TextualEntropy","ComplexClass","LargeClass","LazyClass","RefusedBequest","SpaghettiCode"]
    input_data = np.array(data.drop(useless_input_data, axis=1))[:n]
    scaled_input_data = scaler.fit_transform(input_data)

    useless_output_data = ["ComponentName","nameProject","M_CBO","M_CYCLO","M_DIT","M_ELOC","M_FanIn","M_FanIn_1","M_LCOM","M_LOC","M_LOCNAMM","M_NOA","M_NOC","M_NOM","M_NOMNAMM","M_NOPA","M_PMMM","M_PRB","M_WLOCNAMM","M_WMC","M_WMCNAMM","M_TextualCohesion","M_TextualEntropy"]
    output_data = np.array(data.drop(useless_output_data, axis=1))[:n]
    scaled_output_data = scaler.fit_transform(output_data)

    return scaled_input_data, scaled_output_data

    


