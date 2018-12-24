'''
Helper functions to cross-validate models

Author: Jason Salazer-Adams
Date: 12/24/2018
'''

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate, \
                                    StratifiedKFold

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
    y = df.buy_event
    X = df.iloc[:, 4:]

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
