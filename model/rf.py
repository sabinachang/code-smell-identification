from scipy.io import arff

import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def preprocess(dataset):
    # import dataset
    data_path = str(Path(__file__).parent.parent) + dataset
    data = arff.loadarff(data_path)
    df = pd.DataFrame(data[0])

    # preprocess dataset
    X = df.iloc[:, :-1].values
    Y_data = df.iloc[:, -1].values
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(Y_data)
    X_copy = df.iloc[:, :-1].copy()
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_copy)
    new_X = imputer.transform(X_copy)
    new_X_df = pd.DataFrame(new_X, columns=X_copy.columns, index=X_copy.index)

    # split test and train
    return train_test_split(new_X, y, test_size=0.15, random_state=42)

def predict(best_model, X_train, X_test, y_train, y_test):
    train_acc = str(accuracy_score(y_train, best_model.predict(X_train))) 
    train_f1= str(f1_score(y_train, best_model.predict(X_train)))
    test_acc = str(accuracy_score(y_test, best_model.predict(X_test)))
    test_f1 = str(f1_score(y_test, best_model.predict(X_test)))

    train_line = train_acc + ' ' + train_f1
    test_line = test_acc + ' ' + test_f1

    return train_line, test_line
    
def write_file(train_line, test_line, output):
    file_path = str(Path(__file__).parent.parent) + output
    with open(file_path, 'w') as writer:
        writer.write(train_line + '\n')
        writer.write(test_line)
        
def train_long():
    X_train, X_test, y_train, y_test = preprocess('/dataset/long-method.arff')
    
    # grid search best hyperparameters for decision tree
    depths = np.arange(1, 3)
    estimators = [400, 600,650]
    param_grid =[{ 'n_jobs':[-1], 'bootstrap':[True,False], 'criterion':['gini','entropy'],'max_depth': depths,'random_state':[0], 'n_estimators':estimators,'class_weight':['balanced']},] 
    rnd_clf = RandomForestClassifier()
    print('Training long method smell with random forest. May take a while...')
    grid_search = GridSearchCV(rnd_clf, param_grid, verbose=1, cv=10, scoring='balanced_accuracy',return_train_score=True)
    grid_search.fit(X_train, y_train)

    train_line, test_line = predict(grid_search.best_estimator_, X_train, X_test, y_train, y_test)
    write_file(train_line, test_line, '/output/rf_long.txt')

def train_feature():
    X_train, X_test, y_train, y_test = preprocess('/dataset/feature-envy.arff')

    depths = np.arange(1, 3)
    estimators = [300,400,600]
    param_grid =[{ 'n_jobs':[10], 'bootstrap':[True,False], 'criterion':['gini','entropy'],'max_depth': depths,'random_state':[0], 'n_estimators':estimators,'class_weight':['balanced']},] 
    rnd_clf = RandomForestClassifier()
    print('Training feature envy smell with random forest. May take a while...')
    grid_search = GridSearchCV(rnd_clf, param_grid, verbose=1, cv=10, scoring='balanced_accuracy',return_train_score=True)
    grid_search.fit(X_train, y_train)

    train_line, test_line = predict(grid_search.best_estimator_, X_train, X_test, y_train, y_test)
    write_file(train_line, test_line, '/output/rf_feature.txt')

def train_data():
    X_train, X_test, y_train, y_test = preprocess('/dataset/data-class.arff')

    depths = np.arange(1, 3)
    estimators = [300,400,600]
    param_grid =[{ 'n_jobs':[10], 'bootstrap':[True,False], 'criterion':['gini','entropy'],'max_depth': depths,'random_state':[0], 'n_estimators':estimators,'class_weight':['balanced']},] 
    rnd_clf = RandomForestClassifier()
    print('Training data class smell with random forest. May take a while...')
    grid_search = GridSearchCV(rnd_clf, param_grid, verbose=1, cv=10, scoring='balanced_accuracy',return_train_score=True)
    grid_search.fit(X_train, y_train)

    train_line, test_line = predict(grid_search.best_estimator_, X_train, X_test, y_train, y_test)
    write_file(train_line, test_line, '/output/rf_data.txt')

def train_god():
    X_train, X_test, y_train, y_test = preprocess('/dataset/god-class.arff')

    depths = np.arange(1, 3)
    estimators = [300,400, 600]
    param_grid =[{ 'n_jobs':[-1], 'bootstrap':[True,False], 'criterion':['gini','entropy'],'max_depth': depths,'random_state':[0], 'n_estimators':estimators,'class_weight':['balanced']},] 
    rnd_clf = RandomForestClassifier()
    print('Training god class smell with random forest. May take a while...')
    grid_search = GridSearchCV(rnd_clf, param_grid, verbose=1, cv=10, scoring='balanced_accuracy',return_train_score=True)
    grid_search.fit(X_train, y_train)

    train_line, test_line = predict(grid_search.best_estimator_, X_train, X_test, y_train, y_test)
    write_file(train_line, test_line, '/output/rf_god.txt')
