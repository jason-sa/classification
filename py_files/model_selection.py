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
from sklearn.model_selection import StratifiedKFold, KFold

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
from imblearn.combine import SMOTETomek, SMOTEENN

from collections import Counter
# only uncomment whem comfortable with the warnings
# import warnings
# warnings.filterwarnings("ignore")

RANDOM_STATE = 1234

def create_Xy(df, sample_model=None):
    ''' Performs train test split and applies any sampling.
    
    df: observations data frame to split into X, y
    sample_model: sampling model to apply to the train data set

    returns: X, y, X_train, X_test, y_train, y_test, X_train_orig (no sampling), y_train (no sampling)
    '''
    y = df.buy_event
    X = df.iloc[:,4:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = RANDOM_STATE, stratify=y)

    # up-sample with SMOTE
    # print(f'Prior to re-sample: {Counter(y_train)}')

    X_res = None
    y_res = None
    if sample_model is not None:
        # sm = SMOTE(random_state=RANDOM_STATE)

        X_res, y_res = sample_model.fit_sample(X_train, np.ravel(y_train))

        X_res = pd.DataFrame(X_res)
        y_res = pd.DataFrame(y_res)

        X_res.columns = X_train.columns
        y_res.name = y_train.name

    # print(f'After re-sample: {Counter(y_res)}')

    return X, y, X_res, X_test, y_res, y_test, X_train, y_train

def create_Xy_SMOTETomek(df):
    y = df.buy_event
    X = df.iloc[:,4:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = RANDOM_STATE, stratify=y)

    # up-sample with SMOTE
    # print(f'Prior to re-sample: {Counter(y_train)}')

    sm = SMOTETomek(random_state=RANDOM_STATE)

    X_res, y_res = sm.fit_sample(X_train, np.ravel(y_train))

    X_res = pd.DataFrame(X_res)
    y_res = pd.DataFrame(y_res)

    X_res.columns = X_train.columns
    y_res.name = y_train.name

    # print(f'After re-sample: {Counter(y_res)}')

    return X, y, X_res, X_test, y_res, y_test, X_train, y_train

def create_Xy_SMOTEENN(df):
    y = df.buy_event
    X = df.iloc[:,4:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = RANDOM_STATE, stratify=y)

    # up-sample with SMOTE
    # print(f'Prior to re-sample: {Counter(y_train)}')

    sm = SMOTEENN(random_state=RANDOM_STATE)

    X_res, y_res = sm.fit_sample(X_train, np.ravel(y_train))

    X_res = pd.DataFrame(X_res)
    y_res = pd.DataFrame(y_res)

    X_res.columns = X_train.columns
    y_res.name = y_train.name

    # print(f'After re-sample: {Counter(y_res)}')

    return X, y, X_res, X_test, y_res, y_test, X_train, y_train

def cv_models(X_train, y_train, n_iters=10, scoring='roc_auc', models = [
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
    ''' Trains models utilizing BayesSearchCV with 10 fold CV.

    X_train: training feature space
    y_train: training response space
    n_iters: number of iterations for BayesSearch
    scoring: the socring method to be utilized
    models: models to be trained
    param_choices: parameters to be tuned for each model

    returns dict of BayesSearch CV results for each model
    '''

    skf = KFold(n_splits=10, random_state=RANDOM_STATE, shuffle=True)

    grids = {}
    for model_info, params in zip(models, param_choices):
        name, model = model_info

        print(f'Fitting {name}')
        print()

        grid = BayesSearchCV(
                            model(), 
                            params, 
                            scoring=scoring, 
                            n_iter=n_iters, 
                            cv=skf,
                            n_jobs=5) 

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
    ''' Runs train/test split and models with defaults

    returns dict BayesSearchCV results 
    '''
    obs = utils.load_pickle('features')

    # build X,y and perform train test split
    X, y, X_train, X_test, y_train, t_test, X_train_orig, y_train_orig = create_Xy(obs)

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