from src.data.clean import clean_data
from src.data.data_Import import transform_data_to_df
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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




if __name__ == "__main__":
    standardize_data()
    L2_logistic_regression()
    L2_logistic_regression_newton()
    L1_logistic_regression_saga()
    L2_logistic_regression_saga()
    L1_logistic_regression_liblinear()
    L2_logistic_regression_liblinear()  