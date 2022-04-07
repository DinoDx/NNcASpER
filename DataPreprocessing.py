import pandas as pd
import numpy as np

class DataPreprocessing :

    def __init__(self) -> None:
        pass

    global data 
    data = pd.read_csv("complete_Dataset.csv", sep=",")

    def inputData(self, n) -> np.ndarray:
        useless_input_data = ["ComponentName", "nameProject", "M_TextualCohesion","M_TextualEntropy","ComplexClass","LargeClass","LazyClass","RefusedBequest","SpaghettiCode"]
        input_data = np.array(data.drop(useless_input_data, axis=1))[:n]
        return input_data

    def outputData(self, n) -> np.ndarray:
        useless_output_data = ["ComponentName","nameProject","M_CBO","M_CYCLO","M_DIT","M_ELOC","M_FanIn","M_FanIn_1","M_LCOM","M_LOC","M_LOCNAMM","M_NOA","M_NOC","M_NOM","M_NOMNAMM","M_NOPA","M_PMMM","M_PRB","M_WLOCNAMM","M_WMC","M_WMCNAMM","M_TextualCohesion","M_TextualEntropy"]
        output_data = np.array(data.drop(useless_output_data, axis=1))[:n]
        return output_data

    


