''' Runs different classification algorithms for model selection

Author: Jason Salazer-Adams
Date: 10/24/2018

'''
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

def load_pickle(name):
    return pd.read_pickle(f'../data/{name}.pkl')

def create_Xy(df):
    y = df.buy_event
    X = df.drop(columns='buy_event')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1234)

    return X, y, X_train, X_test, y_train, y_test

def cv_models(X_train, y_train):
    models = [('knn', KNN), 
            ('logistic', LogisticRegression),
            ('tree', DecisionTreeClassifier),
            ('forest', RandomForestClassifier)
            ]

    param_choices = [
        {
            'n_neighbors': range(1, 12)
        },
        {
            'C': np.logspace(-3,6, 12),
            'penalty': ['l1', 'l2']
        },
        {
            'max_depth': [1,2,3,4,5],
            'min_samples_leaf': [3,6,10]
        },
        {
            'n_estimators': [50, 100, 200],
            'max_depth': [1,2,3,4,5],
            'min_samples_leaf': [3,6,10]
        }
    ]

    grids = {}
    for model_info, params in zip(models, param_choices):
        name, model = model_info
        grid = GridSearchCV(model(), params)
        grid.fit(X_train, y_train)
        s = f"{name}: best score: {grid.best_score_}"
        print(s)
        grids[name] = grid
    
    return grid

def run_models():
    obs = load_pickle('observations')

    # build X,y and perform train test split
    X, y, X_train, X_test, y_train, t_test = create_Xy(obs)

    # run the models
    results = cv_models(X_train, y_train)
    
    return results

if __name__ == '__main__':
    run_models()