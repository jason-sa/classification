'''
Trains the best model

Author: Jason Salazer-Adams
Date: 12/24/2018
'''
import pandas as pd
import logging

# modeling
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbPipeline

from src.models.cross_validate import create_Xy

from pathlib import Path
import os
import pickle

RANDOM_STATE = 42


def run_model():
    '''
    Trains the best model, and pickles the model at
    /models/final_model.pkl

    returns dict BayesSearchCV results
    '''
    logger = logging.getLogger(__name__)
    logger.info('Training model')

    project_dir = Path(__file__).resolve().parents[2]
    obs_features_path = os.path.join(project_dir,
                                     'data',
                                     'processed',
                                     'observations_features.csv')
    obs = pd.read_csv(obs_features_path)

    logger.info('Train / test split')
    # perform train test split
    X_train, X_test, y_train, y_test = create_Xy(obs)

    # create model
    pipe = imbPipeline([
                        ('ss', StandardScaler()),
                        ('smote', SMOTE(random_state=RANDOM_STATE,
                                        k_neighbors=6)),
                        ('lm', LogisticRegression(random_state=RANDOM_STATE,
                                                  solver='saga',
                                                  C=1e5,
                                                  penalty='l2',))
                       ])

    logger.info('Fitting training data.')
    pipe.fit(X_train, y_train)

    y_test_probs = pipe.predict_proba(X_test)[:, 1]
    test_auc = metrics.roc_auc_score(y_test, y_test_probs)
    logger.info(f'Test AUC score: {test_auc:.3}')

    logger.info('Pickling model')
    model_path = os.path.join(project_dir, 'models', 'final_model.pkl')
    pickle.dump(pipe, open(model_path, 'wb'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    run_model()
