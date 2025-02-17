from turtle import title
from src.data.clean import clean_data
from src.data.data_Import import transform_data_to_df
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd

data = clean_data()
x_data = data.drop('Result', axis=1)

#standardize data
def standardize_data():
    scaler = StandardScaler()
    Standardize = scaler.fit_transform(x_data)
    standardize_df = pd.DataFrame(Standardize)
    return standardize_df
    #print(standardize_df.head())

#train test split (split data using 75/25)
X = standardize_data()
y = data['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#logistic regression with default parameters including penalty='l2' and solver='lbfgs'
def L2_logistic_regression():
    model = LogisticRegression(random_state=42,penalty='l2',solver='lbfgs')
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("L2 Logistic Regression training score: ",train_score)
    print("L2 Logistic Regression accuracy score: ",accuracy_score(y_test, y_pred))

#logistic regression with parameters including penalty='l2' and solver='newton-cg'
def L2_logistic_regression_newton():
    model = LogisticRegression(random_state=42,penalty='l2',solver='newton-cg')
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("L2 Logistic Regression with Newton-CG Solver training score: ",train_score)
    print("L2 Logistic Regression with Newton-CG Solver accuracy score: ",accuracy_score(y_test, y_pred))


#logistic regression with parameters including penalty='l1' and solver='saga'
def L1_logistic_regression_saga():
    model = LogisticRegression(random_state=42,penalty='l1',solver='saga')
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("L1 Logistic Regression with SAGA Solver training score: ",train_score)
    print("L1 Logistic Regression with SAGA Solver accuracy score: ",accuracy_score(y_test, y_pred))

#logistic regression with parameters including penalty='l2' and solver='saga'
def L2_logistic_regression_saga():
    model = LogisticRegression(random_state=42,penalty='l2',solver='saga')
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("L2 Logistic Regression with SAGA Solver training score: ",train_score)
    print("L2 Logistic Regression with SAGA Solver accuracy score: ",accuracy_score(y_test, y_pred))


#logistic regression with parameters including penalty='l1' and solver='liblinear'
def L1_logistic_regression_liblinear():
    model = LogisticRegression(random_state=42,penalty='l1',solver='liblinear')
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("L1 Logistic Regression with LIBLINEAR Solver training score: ",train_score)
    print("L1 Logistic Regression with LIBLINEAR Solver accuracy score: ",accuracy_score(y_test, y_pred))


#logistic regression with parameters including penalty='l2' and solver='liblinear'
def L2_logistic_regression_liblinear():
    model = LogisticRegression(random_state=42,penalty='l2',solver='liblinear')
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("L2 Logistic Regression with LIBLINEAR Solver training score: ",train_score)
    print("L2 Logistic Regression with LIBLINEAR Solver accuracy score: ",accuracy_score(y_test, y_pred))

#random forest with default parameters
def Random_Forest():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest training score: ",train_score)
    print("Random Forest accuracy score: ",accuracy_score(y_test, y_pred))

#random forest with 50 estimators
def Random_Forest_50estimators():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest 50 estimators training score: ",train_score)
    print("Random Forest 50 estimators accuracy score: ",accuracy_score(y_test, y_pred))

#random forest with 10 estimators
def Random_Forest_10estimators():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest 10 estimators training score: ",train_score)
    print("Random Forest 10 estimators accuracy score: ",accuracy_score(y_test, y_pred))

#random forest with max depth of 5, default is none 
def Random_Forest_MaxDepth():
    model = RandomForestClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest Max Depth 5training score: ",train_score)
    print("Random Forest Max Depth 5accuracy score: ",accuracy_score(y_test, y_pred))

#random forest with min split of 5, default is 2
def Random_Forest_MinSpilt5():
    model = RandomForestClassifier(min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest Min Spilt 5 training score: ",train_score)
    print("Random Forest Min Spilt 5 accuracy score: ",accuracy_score(y_test, y_pred))

#random forest with min split of 2 and max depth of 5
def Random_Forest_Split_Depth():
    model = RandomForestClassifier(min_samples_split=2, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest Min Spilt 2 Max Depth 5 training score: ",train_score)
    print("Random Forest Min Spilt 2 Max Depth 5 accuracy score: ",accuracy_score(y_test, y_pred))

#feature importance from random forest model
def Feature_importance():
    feature_names = x_data.columns
    random_forest_model = RandomForestClassifier(random_state=42)
    random_forest_model.fit(X_train, y_train)
    importances = random_forest_model.feature_importances_
    feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=False) 
    print(feature_df)
    
    # Set figure size
    plt.figure(figsize=(12, 6))
    
    # Create the barplot
    feature_plot = sns.barplot(x='Feature', y='Importance', data=feature_df)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add title and labels
    plt.title('Feature Importance from Random Forest Model')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    return feature_plot


#support vector machine model, default parameters
def SVM_model():
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("SVM training score: ",train_score)
    print("SVM accuracy score: ",accuracy_score(y_test, y_pred))

def SVM_model_balanced():
    model = SVC(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    print("SVM balanced training score: ",train_score)
    print("SVM balanced accuracy score: ",accuracy_score(y_test, y_pred))




if __name__ == "__main__":
    standardize_data()
    L2_logistic_regression()
    L2_logistic_regression_newton()
    L1_logistic_regression_saga()
    L2_logistic_regression_saga()
    L1_logistic_regression_liblinear()
    L2_logistic_regression_liblinear()  
    Random_Forest()
    Random_Forest_50estimators()
    Random_Forest_10estimators()
    Random_Forest_MaxDepth()
    Random_Forest_MinSpilt5()
    Random_Forest_Split_Depth()
    Feature_importance()
    SVM_model()
    SVM_model_balanced()



