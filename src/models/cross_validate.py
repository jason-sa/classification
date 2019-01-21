'''
Helper functions to cross-validate models

Author: Jason Salazer-Adams
Date: 12/24/2018
'''

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, \
                                    StratifiedKFold

from skopt import BayesSearchCV

RANDOM_STATE = 42


def create_Xy(df):
    '''
    Performs train test split and applies any sampling.

    df:
        observations data frame to split into X, y
    sample_model:
        sampling model to apply to the train data set

    returns:
        X_train, X_test, y_train, y_test
    '''
    y = np.array(df.buy_event)
    X = np.array(df.iloc[:, 4:])

    return train_test_split(X, y, random_state=RANDOM_STATE, stratify=y)


def cv_model(X_train, y_train, model):
    '''
    Performs stratified 3-fold cross-validation of the model.

    X_train:
        training feature space
    y_train:
        training response space
    model:
        model or pipeline to cross-validate 3-fold

    returns: sklearn cv dict
        cv results
    '''

    skf = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE)

    cv = cross_validate(model,
                        X_train,
                        y_train,
                        cv=skf,
                        scoring=['accuracy', 'recall', 'precision',
                                 'f1', 'roc_auc'],
                        return_train_score=False,
                        n_jobs=3)

    return cv


def log_scores(cv, m_name):
    '''
    Calculates the average and standard deviation of the classification errors.
    The full list in the return documentation.

    cv: dict
        Dictionary of cv results produced from sklearn cross_validate.

    m_name: string
        Name of the model to use as the index

    return: DataFrame
        DataFrame (model name, metrics). Metrics currently implemented
        and measured on the test fold are,
        - accuracy
        - precision
        - recall
        - F1
        - AUC
    '''
    measures = []
    for k, v in cv.items():
        if 'test' in k:
            measures.append(v.mean())
            measures.append(v.std())
    measures = np.array(measures)

    return pd.DataFrame(data=measures.reshape(1, -1),
                        columns=['avg_accuracy', 'std_accuracy',
                                 'avg_precision', 'std_precision',
                                 'avg_recall', 'std_recall',
                                 'avg_f1', 'std_f1',
                                 'avg_auc', 'std_auc'],
                        index=[m_name])


def bayes_search(X_train, y_train, model, search_params):
    '''
    Performs a BayesSearchCV on the provided model and search parameters.

    X_train:
        feature space array
    y_train:
        response array
    model:
        sklearn model of pipeline to be tuned
    search_params:
        dictionary of search parameters

    return:
        skopt cv results (dict)
    '''
    skf = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE)

    bayes_params = {
        'estimator': model,
        'scoring': 'roc_auc',
        'search_spaces': search_params,
        'n_iter': 50,
        'cv': skf,
        'n_jobs': 3,
        'verbose': 1
    }

    search_cv = BayesSearchCV(**bayes_params)
    search_cv_results = search_cv.fit(X_train,
                                      y_train)

    return search_cv_results


def print_score_progress(optim_results):
    '''
    Callback for BayesSearchCV
    Prints the best score, current score, and current iteration
    '''
    current_results = pd.DataFrame(optim_results.cv_results_)
    print(f'Iteration: {current_results.shape[0]}')
    print(f'Current AUC: ' +
          f'{current_results.tail(1).mean_test_score.values[0]:.6f}')
    print(f'Best AUC: {optim_results.best_score_:.6f}')
    print()


# def save_best_estimator(optim_results):
#     '''
#     BayesSearchCV callback
#     Saves best estimator
#     '''
#     current_results = pd.DataFrame(search_cv.cv_results_)
#     best_score = search_cv.best_score_
#     current_score = current_results.tail(1).mean_test_score.values[0]
#     if current_score == best_score:
#         model = f'tuned_models/{model_name}_{best_score:.6f}'
#         print(f'Saving: {model}')
#         print()
#         utils.save(search_cv, model)
