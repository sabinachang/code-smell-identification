from scipy.io import arff

import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
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

def train_linear_feature():
    X_train, X_test, y_train, y_test = preprocess('/dataset/feature-envy.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100],  'dual': [False],'random_state':[False]}]
    grid = GridSearchCV(LinearSVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcl_feature.txt')

def train_poly_feature():
    X_train, X_test, y_train, y_test = preprocess('/dataset/feature-envy.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['poly'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcp_feature.txt')   

def train_sigmoid_feature():
    X_train, X_test, y_train, y_test = preprocess('/dataset/feature-envy.arff')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['sigmoid'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcs_feature.txt') 

def train_rbf_feature():
    X_train, X_test, y_train, y_test = preprocess('/dataset/feature-envy.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['rbf'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcr_feature.txt') 

def train_linear_data():
    X_train, X_test, y_train, y_test = preprocess('/dataset/data-class.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100],  'dual': [False],'random_state':[False]}]    
    grid = GridSearchCV(LinearSVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcl_data.txt') 

def train_poly_data():
    X_train, X_test, y_train, y_test = preprocess('/dataset/data-class.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['poly'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcp_data.txt') 

def train_rbf_data():
    X_train, X_test, y_train, y_test = preprocess('/dataset/data-class.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['rbf'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcr_data.txt') 

def train_sigmoid_data():
    X_train, X_test, y_train, y_test = preprocess('/dataset/data-class.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['rbf'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcs_data.txt') 

def train_linear_god():
    X_train, X_test, y_train, y_test = preprocess('/dataset/god-class.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100],  'dual': [False],'random_state':[False]}]
    grid = GridSearchCV(LinearSVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcl_god.txt') 

def train_poly_god():
    X_train, X_test, y_train, y_test = preprocess('/dataset/god-class.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['rbf'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcp_god.txt') 

def train_rbf_god():
    X_train, X_test, y_train, y_test = preprocess('/dataset/god-class.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['rbf'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcr_god.txt') 

def train_sigmoid_god():
    X_train, X_test, y_train, y_test = preprocess('/dataset/god-class.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['sigmoid'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcs_god.txt') 

def train_linear_long():
    X_train, X_test, y_train, y_test = preprocess('/dataset/long-method.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100],  'dual': [False],'random_state':[False]}]
    grid = GridSearchCV(LinearSVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcl_long.txt') 

def train_poly_long():
    X_train, X_test, y_train, y_test = preprocess('/dataset/long-method.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['poly'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcp_long.txt') 

def train_rbf_long():
    X_train, X_test, y_train, y_test = preprocess('/dataset/long-method.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['rbf'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcr_long.txt') 

def train_sigmoid_long():
    X_train, X_test, y_train, y_test = preprocess('/dataset/long-method.arff')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    param_grid = [{'C': [0.001,0.1,1, 10,100], 'gamma': [1,0.1,0.01,0.001, 'auto'],'kernel': ['sigmoid'], 'probability':[False]}]
    grid = GridSearchCV(SVC(),param_grid,refit=True,cv=10, scoring='f1',verbose=1,return_train_score=True)
    grid.fit(X_train_scaled,y_train)

    X_test_scaled = scaler.fit_transform(X_test)
    train_line, test_line = predict(grid.best_estimator_, X_train_scaled, X_test_scaled, y_train, y_test)
    write_file(train_line, test_line, '/output/svcs_long.txt') 