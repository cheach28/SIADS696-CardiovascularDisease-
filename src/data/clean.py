import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_Import import transform_data_to_df

#remove error entries from data (heart rate over 1000 is not possible)
def clean_data():
    data = transform_data_to_df()
    clean_df = data[data['Heart rate'] < 500]
    return clean_df

if __name__ == "__main__":
    clean_data()

    
   

    
    