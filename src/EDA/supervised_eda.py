
from src.data.clean import clean_data
from src.data.data_Import import transform_data_to_df
import matplotlib.pyplot as plt
import seaborn as sns

def raw_eda():

    data = transform_data_to_df()

    #show basic characterstics of the data
    print(data.describe())

    # Set figure size with adequate height for labels
    fig, ax = plt.subplots(figsize=(14, 10))

    # Prepare the data for vital signs
    features = data[['Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure','Blood sugar']]
    features = features.rename(columns={
        'Age': 'Age (Years)', 
        'Heart rate': 'Heart rate (BPM)', 
        'Systolic blood pressure': 'Systolic blood pressure (mmHg)', 
        'Diastolic blood pressure': 'Diastolic blood pressure (mmHg)', 
        'Blood sugar': 'Blood sugar (mg/dL)'
    })

    # Create the boxplot with grey outline
    boxplot_data = features.boxplot(ax=ax, return_type='dict', boxprops=dict(color='grey'), 
                                  meanprops=dict(color='pink'), medianprops=dict(color='black'),
                                  whiskerprops=dict(color='black'))

    # Customize the plot
    plt.ylim(-1, 500)
    plt.grid(False)
    plt.title('Distribution of Age and Vital Sign Features', pad=20)

    # Add value annotations
    for i, box in enumerate(boxplot_data['boxes']):
        # Get the statistics
        median = boxplot_data['medians'][i].get_ydata()[0]
        upper_quartile = box.get_path().vertices[3][1]
        lower_quartile = box.get_path().vertices[0][1]
        
        # Add text annotations
        ax.text(i + 1, median, f'{int(median)}', 
                ha='center', va='center', fontsize=8, 
                color='white', bbox=dict(facecolor='black', alpha=0.5))
        ax.text(i + 1, upper_quartile, f'{int(upper_quartile)}', 
                ha='center', va='bottom', fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.5))
        ax.text(i + 1, lower_quartile - 0.2, f'{int(lower_quartile)}', 
                ha='center', va='top', fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.5))

    # Adjust x-axis labels
    ax.tick_params(axis='x', rotation=90, labelsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.show()

    # Create figure for CK-MB plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create the boxplot with matching style for CK-MB
    ck_mb = data[['CK-MB']]
    boxplot_data = ck_mb.boxplot(ax=ax, return_type='dict', boxprops=dict(color='grey'), 
                                meanprops=dict(color='pink'), medianprops=dict(color='black'),
                                whiskerprops=dict(color='black'))
    
    # Customize the plot
    plt.ylim(-1, 75)
    plt.yticks(range(0, 76, 5))  # Create ticks from 0 to 75 in steps of 5
    plt.grid(False)
    plt.title('Distribution of Creatine Kinase-MB Laboratory Blood Serum Test', pad=20)
    plt.ylabel('CK-MB (IU/L)')

    # Add value annotations
    for i, box in enumerate(boxplot_data['boxes']):
        # Get the statistics
        median = boxplot_data['medians'][i].get_ydata()[0]
        upper_quartile = box.get_path().vertices[3][1]
        lower_quartile = box.get_path().vertices[0][1]
        
        # Add text annotations
        ax.text(i + 1, median, f'{int(median)}', 
                ha='center', va='center', fontsize=8, 
                color='white', bbox=dict(facecolor='black', alpha=0.5))
        ax.text(i + 1, upper_quartile, f'{int(upper_quartile)}', 
                ha='center', va='bottom', fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.5))
        ax.text(i + 1, lower_quartile - 0.2, f'{int(lower_quartile)}', 
                ha='center', va='top', fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.5))

    # Adjust x-axis labels
    ax.tick_params(axis='x', rotation=90, labelsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.show()

    # Create figure for Troponin plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create the boxplot with matching style for Troponin
    trop = data[['Troponin']]
    boxplot_data = trop.boxplot(ax=ax, return_type='dict', boxprops=dict(color='grey'), 
                               meanprops=dict(color='pink'), medianprops=dict(color='black'),
                               whiskerprops=dict(color='black'))
    
    # Customize the plot
    plt.ylim(-0.1, 2)
    plt.yticks([x/10 for x in range(0, 21, 2)])  # Create ticks from 0 to 2 in steps of 0.2
    plt.grid(False)
    plt.title('Distribution of Troponin Laboratory Blood Serum Test', pad=20)
    plt.ylabel('Troponin (ng/L)')

    # Add value annotations
    for i, box in enumerate(boxplot_data['boxes']):
        # Get the statistics
        median = boxplot_data['medians'][i].get_ydata()[0]
        upper_quartile = box.get_path().vertices[3][1]
        lower_quartile = box.get_path().vertices[0][1]
        
        # Add text annotations with 3 decimal places for Troponin values
        ax.text(i + 1, median, f'{median:.3f}', 
                ha='center', va='center', fontsize=8, 
                color='white', bbox=dict(facecolor='black', alpha=0.5))
        ax.text(i + 1, upper_quartile, f'{upper_quartile:.3f}', 
                ha='center', va='bottom', fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.5))
        ax.text(i + 1, lower_quartile - 0.02, f'{lower_quartile:.3f}', 
                ha='center', va='top', fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.5))

    # Adjust x-axis labels
    ax.tick_params(axis='x', rotation=90, labelsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.show()


def clean_eda():
    clean = clean_data()

    print(clean.describe())

    # Set figure size with adequate height for labels
    fig, ax = plt.subplots(figsize=(14, 10))

    # Prepare the data
    clean_features = clean[['Age', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure','Blood sugar']]
    clean_features = clean_features.rename(columns={
        'Age': 'Age (Years)', 
        'Heart rate': 'Heart rate (BPM)', 
        'Systolic blood pressure': 'Systolic blood pressure (mmHg)', 
        'Diastolic blood pressure': 'Diastolic blood pressure (mmHg)', 
        'Blood sugar': 'Blood sugar (mg/dL)'
    })

    # Create the boxplot with grey outline and pink mean line
    boxplot_data = clean_features.boxplot(ax=ax, return_type='dict', boxprops=dict(color='grey'), meanprops=dict(color='pink'), medianprops=dict(color='black'),whiskerprops=dict(color='black'))

    # Customize the plot
    plt.ylim(-1, 400)
    plt.grid(False)
    plt.title('Distribution of Age and Vital Sign Features', pad=20)

    # Add value annotations
    for i, box in enumerate(boxplot_data['boxes']):
        # Get the statistics
        median = boxplot_data['medians'][i].get_ydata()[0]
        upper_quartile = box.get_path().vertices[3][1]
        lower_quartile = box.get_path().vertices[0][1]
        
        # Add text annotations
        ax.text(i + 1, median, f'{int(median)}', 
                ha='center', va='center', fontsize=8, 
                color='white', bbox=dict(facecolor='black', alpha=0.5))
        ax.text(i + 1, upper_quartile, f'{int(upper_quartile)}', 
                ha='center', va='bottom', fontsize=8,color= 'white',bbox=dict(facecolor='red', alpha=0.5))
        ax.text(i + 1, lower_quartile - 0.2, f'{int(lower_quartile)}', 
                ha='center', va='top', fontsize=8,color= 'white',bbox=dict(facecolor='red', alpha=0.5))

    # Adjust x-axis labels
    ax.tick_params(axis='x', rotation=90, labelsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.show()



    # Create figure for CK-MB plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create the boxplot with matching style
    ck_mb = clean[['CK-MB']]
    boxplot_data = ck_mb.boxplot(ax=ax, return_type='dict', boxprops=dict(color='grey'), 
                                meanprops=dict(color='pink'), medianprops=dict(color='black'),
                                whiskerprops=dict(color='black'))
    
    # Customize the plot
    plt.ylim(-1, 75)
    plt.yticks(range(0, 76, 5))  # Create ticks from 0 to 75 in steps of 5
    plt.grid(False)
    plt.title('Distribution of Creatine Kinase-MB Laboratory Blood Serum Test', pad=20)
    plt.ylabel('CK-MB (IU/L)')

    # Add value annotations
    for i, box in enumerate(boxplot_data['boxes']):
        # Get the statistics
        median = boxplot_data['medians'][i].get_ydata()[0]
        upper_quartile = box.get_path().vertices[3][1]
        lower_quartile = box.get_path().vertices[0][1]
        
        # Add text annotations
        ax.text(i + 1, median, f'{int(median)}', 
                ha='center', va='center', fontsize=8, 
                color='white', bbox=dict(facecolor='black', alpha=0.5))
        ax.text(i + 1, upper_quartile, f'{int(upper_quartile)}', 
                ha='center', va='bottom', fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.5))
        ax.text(i + 1, lower_quartile - 0.2, f'{int(lower_quartile)}', 
                ha='center', va='top', fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.5))

    # Adjust x-axis labels
    ax.tick_params(axis='x', rotation=90, labelsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.show()

    # Create figure for Troponin plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create the boxplot with matching style
    trop = clean[['Troponin']]
    boxplot_data = trop.boxplot(ax=ax, return_type='dict', boxprops=dict(color='grey'), 
                               meanprops=dict(color='pink'), medianprops=dict(color='black'),
                               whiskerprops=dict(color='black'))
    
    # Customize the plot
    plt.ylim(-0.1, 2)
    plt.yticks([x/10 for x in range(0, 21, 2)])  # Create ticks from 0 to 2 in steps of 0.2
    plt.grid(False)
    plt.title('Distribution of Troponin Laboratory Blood Serum Test', pad=20)
    plt.ylabel('Troponin (ng/L)')

    # Add value annotations
    for i, box in enumerate(boxplot_data['boxes']):
        # Get the statistics
        median = boxplot_data['medians'][i].get_ydata()[0]
        upper_quartile = box.get_path().vertices[3][1]
        lower_quartile = box.get_path().vertices[0][1]
        
        # Add text annotations with 3 decimal places for Troponin values
        ax.text(i + 1, median, f'{median:.3f}', 
                ha='center', va='center', fontsize=8, 
                color='white', bbox=dict(facecolor='black', alpha=0.5))
        ax.text(i + 1, upper_quartile, f'{upper_quartile:.3f}', 
                ha='center', va='bottom', fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.5))
        ax.text(i + 1, lower_quartile - 0.02, f'{lower_quartile:.3f}', 
                ha='center', va='top', fontsize=8, color='white',
                bbox=dict(facecolor='red', alpha=0.5))

    # Adjust x-axis labels
    ax.tick_params(axis='x', rotation=90, labelsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    plt.show()





if __name__ == "__main__":
    raw_eda()
    clean_eda()
    
