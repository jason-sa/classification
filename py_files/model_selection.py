''' Runs different classification algorithms for model selection

Author: Jason Salazer-Adams
Date: 10/24/2018

'''
import pandas as pd 
import numpy as np
import pickle

import utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler

from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV

from imblearn.over_sampling import SMOTE

from collections import Counter
# only uncomment whem comfortable with the warnings
# import warnings
# warnings.filterwarnings("ignore")

RANDOM_STATE = 1234

def create_Xy(df):
    y = df.buy_event
    X = df.iloc[:,4:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = RANDOM_STATE, stratify=y)

    # up-sample with SMOTE
    # print(f'Prior to re-sample: {Counter(y_train)}')

    sm = SMOTE(random_state=RANDOM_STATE)

    X_res, y_res = sm.fit_sample(X_train, np.ravel(y_train))

    X_res = pd.DataFrame(X_res)
    y_res = pd.DataFrame(y_res)

    X_res.columns = X_train.columns
    y_res.name = y_train.name

    # print(f'After re-sample: {Counter(y_res)}')

    return X, y, X_res, X_test, y_res, y_test

def cv_models(X_train, y_train, n_iters=10, models = [
            ('Gradient Boost', GradientBoostingClassifier), 
            ('Random Forest', RandomForestClassifier)
            ], param_choices = [
        {
            'loss': Categorical(['deviance', 'exponential']),
            'learning_rate': Real(1e-2, 1),
            'n_estimators': Integer(100, 500),
            'random_state': [RANDOM_STATE]
        },
        {
            'n_estimators': Integer(100,500),
            'max_depth': Integer(1, 5),
            'random_state': [RANDOM_STATE]
        }
    ]):

    skf = StratifiedKFold(n_splits=10, random_state=RANDOM_STATE)

    grids = {}
    for model_info, params in zip(models, param_choices):
        name, model = model_info

        print(f'Fitting {name}')
        print()

        grid = BayesSearchCV(
                            model(), 
                            params, 
                            scoring='roc_auc', 
                            n_iter=n_iters, 
                            cv=skf,
                            n_jobs=10) 

        if name == 'logistic':
            ssX = StandardScaler()
            X_train_scaled = ssX.fit_transform(X_train)
            grid.fit(X_train_scaled, y_train)
        else:
            grid.fit(X_train, np.ravel(y_train))

        # s = f"{name}: best AUC: {grid.best_score_}"
        # print(s)
        grids[name] = grid
    
    return grids

def run_models():
    obs = utils.load_pickle('features')

    # build X,y and perform train test split
    X, y, X_train, X_test, y_train, t_test = create_Xy(obs)

    # run the models
    results = cv_models(X_train, y_train)
    
    return results

if __name__ == '__main__':
    results = run_models()
    # print(results.shape)
    features = ['view_count', 'session_length', 'item_views', 'add_to_cart_count',
       'transaction_count', 'avg_avail']

    for k, v in results.items():
        print(f'{k} Summary')
        print()
        print(f'Best out-of-sample AUC score: {v.best_score_}')
        print()
        print(f'Feature Importance:')
        for f, i in zip(features, v.best_estimator_.feature_importances_):
            print(f'{f}: \t {i:.2%}')
        print()
        print('----------------------------')
        pickle.dump( v.best_estimator_, open( f"../data/best_{k}_model.pkl", "wb" ) )


    
    ### need to pickle the best model

    # for k, v in results.items():
    #     print(f'{k} best F1 score: {v.best_score_}')