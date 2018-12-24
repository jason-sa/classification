'''
Trains the best model

Author: Jason Salazer-Adams
Date: 12/24/2018
'''
import pandas as pd

from sklearn.linear_model import LogisticRegression
from src.models.cross_validate import create_Xy

from pathlib import Path
import os
import pickle


def run_model():
    '''
    Trains the best model, and pickles the model at
    /models/final_model.pkl

    returns dict BayesSearchCV results
    '''
    project_dir = Path(__file__).resolve().parents[2]
    obs_features_path = os.path.join(project_dir,
                                     'data',
                                     'processed',
                                     'observations_features.csv')
    obs = pd.read_csv(obs_features_path)

    # perform train test split
    X_train, X_test, y_train, y_test = create_Xy(obs)

    # create model
    lg = LogisticRegression(solver='lbfgs')
    lg.fit(X_train, y_train)

    model_path = os.path.join(project_dir, 'models', 'final_model.pkl')
    pickle.dump(lg, open(model_path, 'wb'))


if __name__ == '__main__':
    run_model()