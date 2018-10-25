''' Runs different classification algorithms for model selection

Author: Jason Salazer-Adams
Date: 10/24/2018

'''
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV

# only uncomment whem comfortable with the warnings
import warnings
warnings.filterwarnings("ignore")

def load_pickle(name):
    return pd.read_pickle(f'../data/{name}.pkl')

def create_Xy(df):
    y = df.buy_event
    X = df.iloc[:,4:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1234)

    return X, y, X_train, X_test, y_train, y_test

def cv_models(X_train, y_train):
    models = [('logistic', LogisticRegression),
            ('tree', DecisionTreeClassifier),
            ('forest', RandomForestClassifier)
            ]

    param_choices = [
        {
            'C': Real(1e-3, 1e6),
            'penalty': Categorical(['l1', 'l2'])
        },
        {
            'max_depth': Integer(1,5),
            'min_samples_leaf': Integer(3,10)
        },
        {
            'n_estimators': Integer(50,200),
            'max_depth': Integer(1, 5),
            'min_samples_leaf': Integer(3, 10)
        }
    ]

    grids = {}
    for model_info, params in zip(models, param_choices):
        name, model = model_info
        grid = BayesSearchCV(model(), params, scoring='f1', n_iter=20)

        if name == 'logistic':
            ssX = StandardScaler()
            X_train_scaled = ssX.fit_transform(X_train)
            grid.fit(X_train_scaled, y_train)
        else:
            grid.fit(X_train, y_train)

        # s = f"{name}: best score: {grid.best_score_}"
        # print(s)
        grids[name] = grid
    
    return grids

def run_models():
    obs = load_pickle('features')

    # build X,y and perform train test split
    X, y, X_train, X_test, y_train, t_test = create_Xy(obs)

    # run the models
    results = cv_models(X_train, y_train)
    
    return results

if __name__ == '__main__':
    results = run_models()
    for k, v in results.items():
        print(f'{k} best F1 score: {v.best_score_}')