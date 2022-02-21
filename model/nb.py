from scipy.io import arff

import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

def train_feature():
    X_train, X_test, y_train, y_test = preprocess('/dataset/feature-envy.arff')
    
    gnb = GaussianNB()
    print('Training feature envy smell with naive bayes...')
    gnb.fit(X_train, y_train)

    train_line, test_line = predict(gnb, X_train, X_test, y_train, y_test)
    write_file(train_line, test_line, '/output/nb_feature.txt')

def train_god():
    X_train, X_test, y_train, y_test = preprocess('/dataset/god-class.arff')
    
    gnb = GaussianNB()
    print('Training god class smell with naive bayes...')
    gnb.fit(X_train, y_train)

    train_line, test_line = predict(gnb, X_train, X_test, y_train, y_test)
    write_file(train_line, test_line, '/output/nb_god.txt')

def train_long():
    X_train, X_test, y_train, y_test = preprocess('/dataset/long-method.arff')
    
    gnb = GaussianNB()
    print('Training long method smell with naive bayes...')
    gnb.fit(X_train, y_train)

    train_line, test_line = predict(gnb, X_train, X_test, y_train, y_test)
    write_file(train_line, test_line, '/output/nb_long.txt')

def train_data():
    X_train, X_test, y_train, y_test = preprocess('/dataset/data-class.arff')

    gnb = GaussianNB()
    print('Training data class smell with naive bayes...')
    gnb.fit(X_train, y_train)
    
    train_line, test_line = predict(gnb, X_train, X_test, y_train, y_test)
    write_file(train_line, test_line, '/output/nb_data.txt')
