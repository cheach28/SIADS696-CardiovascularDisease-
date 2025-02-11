
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data.data_Import import transform_data_to_df
import matplotlib.pyplot as plt
import seaborn as sns

def raw_eda():

    data = transform_data_to_df()

    #show basic characterstics of the data
    print(data.describe())

    #visualize data
    #features include basic human vital measurements including age, hr, blood pressure and blood sugar
    features = data[['Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure','Blood sugar']]
    features.boxplot(figsize=(14, 8))
    plt.xticks(rotation=45)
    plt.ylim(-1, 500)
    plt.show()

    #CK-MB and Troponin are visualized separately as they are lab tests and measured in smaller values 
    ck_mb = data[['CK-MB']]
    ck_mb.boxplot(figsize=(14, 8))
    plt.ylim(-1, 100)
    plt.show()

    trop = data[['Troponin']]
    trop.boxplot(figsize=(14, 8))
    plt.ylim(-0.1, 2)
    plt.show()







if __name__ == "__main__":
    raw_eda()


  


