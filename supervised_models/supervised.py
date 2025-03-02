from src.data.clean import clean_data
from src.data.data_Import import transform_data_to_df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, mean_absolute_error,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


#import data from cleaned data 
data = clean_data()
x_data = data.drop('Result', axis=1)

#standardize the data
def standardize_data():
    scaler = StandardScaler()
    Standardize = scaler.fit_transform(x_data)
    return Standardize  

# Split data into train, validation, and test sets (60/20/20) to avoid using train data for both grid search and kfold evaluation
X = standardize_data()  
y = (data['Result'] == 'positive').astype(int)  # Maps 'positive' to 1 and 'negative' to 0 

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Second split: separate validation set from remaining training data, training data 60 
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Convert to numpy arrays to avoid errors 
X_train = np.asarray(X_train)
X_val = np.asarray(X_val)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
y_test = np.asarray(y_test)

# find best hyperparameters using grid search cross validation for logistic regression, random forest and support vector machine
# using k-fold cross validation with 5 folds for evaluation after finding hyperparameters

# Grid search cross validation with random forest
def cv_grid_random_forest():
    # Define the model and hyperparameter options
    model = RandomForestClassifier(random_state=42)
    model_parameters = [{
        'n_estimators': list(range(50,200,50)), 
        'max_depth': list(range(2,10,2)),
        'min_samples_split': list(range(2,10,2))
    }]
    
    # Perform grid search cross validation on training data with 5-fold cross validation
    grid_search = GridSearchCV(
        model, 
        model_parameters, 
        cv=5, 
        scoring=['accuracy', 'neg_mean_absolute_error', 'roc_auc'],  
        refit='accuracy',  # use accuracy score to select best model
        return_train_score=True,
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Print random forest grid search cross validation results
    print('\nRandom Forest Grid Search Results:')
    print(f'Best parameters: {grid_search.best_params_}')
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Create k-fold for random forest cross validation on validation set 
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store scores for each fold
    fold_metrics = {
        'accuracy': [],
        'mae': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    # Perform k-fold cross validation with the best random forest model
    print('\nPer-fold Random Forest Scores:')
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):
        # Split data
        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Train and predict
        best_model.fit(X_fold_train, y_fold_train)
        y_fold_pred = best_model.predict(X_fold_val)
        y_fold_proba = best_model.predict_proba(X_fold_val)[:, 1]
        
        # Calculate scores
        fold_acc = accuracy_score(y_fold_val, y_fold_pred)
        fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
        fold_precision = precision_score(y_fold_val, y_fold_pred)
        fold_recall = recall_score(y_fold_val, y_fold_pred)
        fold_f1 = f1_score(y_fold_val, y_fold_pred)
        fold_roc_auc = roc_auc_score(y_fold_val, y_fold_proba)
        
        # Store scores
        fold_metrics['accuracy'].append(fold_acc)
        fold_metrics['mae'].append(fold_mae)
        fold_metrics['precision'].append(fold_precision)
        fold_metrics['recall'].append(fold_recall)
        fold_metrics['f1'].append(fold_f1)
        fold_metrics['roc_auc'].append(fold_roc_auc)
        
        print(f'\nFold {fold}:')
        print(f'Accuracy:  {fold_acc:.3f}')
        print(f'MAE:       {fold_mae:.3f}')
        print(f'Precision: {fold_precision:.3f}')
        print(f'Recall:    {fold_recall:.3f}')
        print(f'F1 Score:  {fold_f1:.3f}')
        print(f'ROC AUC:   {fold_roc_auc:.3f}')
    
    # Print random forest cross validation summary
    print('\nRandom Forest cross validation summary (mean +/- std):')
    for metric, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f'{metric.upper():9s}: {mean_val:.3f} (± {std_val:.3f})')
    
    # Final evaluation on test set
    best_model.fit(X_train, y_train)  
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    print('\nRandom Forest Final Test Set Scores:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
    print(f'MAE: {mean_absolute_error(y_test, y_pred):.3f}')
    print(f'Precision: {precision_score(y_test, y_pred):.3f}')
    print(f'Recall: {recall_score(y_test, y_pred):.3f}')
    print(f'F1 Score: {f1_score(y_test, y_pred):.3f}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}')
    
    print('\nRandom Forest Classification Report:')
    print(classification_report(y_test, y_pred))
    
    return best_model

def cv_grid_svm():
    # Define the model and hyperparameter options
    model = SVC(random_state=42, probability=True)  
    model_parameters = [{
        'kernel': ['linear', 'rbf'],  
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto'] 
    }]
    
    # Perform grid search cross validation on training data
    grid_search = GridSearchCV(
        model, 
        model_parameters, 
        cv=5, 
        scoring=['accuracy', 'neg_mean_absolute_error', 'roc_auc'],
        refit='accuracy',  
        return_train_score=True,
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Print SVM grid search results
    print('\nSVM Grid Search Results:')
    print(f'Best SVM hyperparameters: {grid_search.best_params_}')
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Create k-fold for SVM cross validation using validation data
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store scores for each fold
    fold_metrics = {
        'accuracy': [],
        'mae': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    # Perform k-fold cross validation with the best model
    print('\nSVM Per-fold Scores:')
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):
        # Split data
        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Train and predict
        best_model.fit(X_fold_train, y_fold_train)
        y_fold_pred = best_model.predict(X_fold_val)
        y_fold_proba = best_model.predict_proba(X_fold_val)[:, 1]
        
        # Calculate scores
        fold_acc = accuracy_score(y_fold_val, y_fold_pred)
        fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
        fold_precision = precision_score(y_fold_val, y_fold_pred)
        fold_recall = recall_score(y_fold_val, y_fold_pred)
        fold_f1 = f1_score(y_fold_val, y_fold_pred)
        fold_roc_auc = roc_auc_score(y_fold_val, y_fold_proba)
        
        # Store scores
        fold_metrics['accuracy'].append(fold_acc)
        fold_metrics['mae'].append(fold_mae)
        fold_metrics['precision'].append(fold_precision)
        fold_metrics['recall'].append(fold_recall)
        fold_metrics['f1'].append(fold_f1)
        fold_metrics['roc_auc'].append(fold_roc_auc)
        
        print(f'\nFold {fold}:')
        print(f'Accuracy:  {fold_acc:.3f}')
        print(f'MAE:       {fold_mae:.3f}')
        print(f'Precision: {fold_precision:.3f}')
        print(f'Recall:    {fold_recall:.3f}')
        print(f'F1 Score:  {fold_f1:.3f}')
        print(f'ROC AUC:   {fold_roc_auc:.3f}')
    
    # Print SVM cross validation summary
    print('\nSVM cross validation summary (mean +/- std):')
    for metric, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f'{metric.upper():9s}: {mean_val:.3f} (± {std_val:.3f})')
    
    # Final evaluation on test set
    best_model.fit(X_train, y_train)  
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    print('\nSVM Final Test Set Scores:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
    print(f'MAE: {mean_absolute_error(y_test, y_pred):.3f}')
    print(f'Precision: {precision_score(y_test, y_pred):.3f}')
    print(f'Recall: {recall_score(y_test, y_pred):.3f}')
    print(f'F1 Score: {f1_score(y_test, y_pred):.3f}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}')
    
    print('\nSVM Classification Report:')
    print(classification_report(y_test, y_pred))
    
    return best_model  


#grid search cross validation with logistic regression model
def cv_grid_logistic_regression():

    # Define the model and hyperparameter options
    model = LogisticRegression(random_state=42, max_iter=1000)
    model_parameters = [{
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }, {
        'solver': ['lbfgs', 'saga'],
        'penalty': [None],
        'C': [1.0]  # C is ignored when penalty is None otherwise error 
    }, {
        'solver': ['saga'],
        'penalty': ['l1'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }]
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # find best hyperparameters using grid search cross validation
    grid_search = GridSearchCV(
        model, 
        model_parameters, 
        cv=kfold, 
        scoring=['accuracy', 'neg_mean_absolute_error', 'roc_auc'],
        refit='accuracy',
        return_train_score=True,
        verbose=0,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Print logistic regression grid search results
    print('\nLogistic Regression Grid Search Results:')
    print(f'Logistic Regression Best parameters: {grid_search.best_params_}')
    
    # Get best model 
    best_params = grid_search.best_params_
    best_model = LogisticRegression(C=1.0, penalty=None, solver='lbfgs', random_state=42, max_iter=1000)
    
    # k-fold CV with the best model 
    print('\nPer-fold Logistic Regression Scores:')
    fold_metrics = {
        'accuracy': [],
        'mae': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):
        # Split data
        X_fold_train = X_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        # Train and predict
        best_model.fit(X_fold_train, y_fold_train)
        y_fold_pred = best_model.predict(X_fold_val)
        
        # Calculate scores
        fold_acc = accuracy_score(y_fold_val, y_fold_pred)
        fold_mae = mean_absolute_error(y_fold_val, y_fold_pred)
        fold_precision = precision_score(y_fold_val, y_fold_pred)
        fold_recall = recall_score(y_fold_val, y_fold_pred)
        fold_f1 = f1_score(y_fold_val, y_fold_pred)
        fold_roc_auc = roc_auc_score(y_fold_val, best_model.predict_proba(X_fold_val)[:, 1])
        
        # Store scores for each fold
        fold_metrics['accuracy'].append(fold_acc)
        fold_metrics['mae'].append(fold_mae)
        fold_metrics['precision'].append(fold_precision)
        fold_metrics['recall'].append(fold_recall)
        fold_metrics['f1'].append(fold_f1)
        fold_metrics['roc_auc'].append(fold_roc_auc)
        
        print(f'\nFold {fold}:')
        print(f'Accuracy:  {fold_acc:.3f}')
        print(f'MAE:       {fold_mae:.3f}')
        print(f'Precision: {fold_precision:.3f}')
        print(f'Recall:    {fold_recall:.3f}')
        print(f'F1 Score:  {fold_f1:.3f}')
        print(f'ROC AUC:   {fold_roc_auc:.3f}')
    
    print('\nLogistic Regression cross validation summary (mean +/- std):')
    for metric, values in fold_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f'{metric.upper():9s}: {mean_val:.3f} (± {std_val:.3f})')
    
    # Final evaluation on test set
    best_model.fit(X_train, y_train)  
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    print('\nFinal Logistic Regression Test Set Scores:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')
    print(f'MAE: {mean_absolute_error(y_test, y_pred):.3f}')
    print(f'Precision: {precision_score(y_test, y_pred):.3f}')
    print(f'Recall: {recall_score(y_test, y_pred):.3f}')
    print(f'F1 Score: {f1_score(y_test, y_pred):.3f}')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred_proba):.3f}')
    
    print('\nLogistic Regression Classification Report:')
    print(classification_report(y_test, y_pred))
    
    return best_model




if __name__ == "__main__":
    standardize_data()
    cv_grid_random_forest()
    cv_grid_svm()
    cv_grid_logistic_regression()


