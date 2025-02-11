import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def transform_data_to_df():
    # supervised dataset
    return pd.read_csv("src/data/Medicaldataset.csv")
    