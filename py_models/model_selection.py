''' Runs different classification algorithms for model selection

Author: Jason Salazer-Adams
Date: 10/24/2018

'''
import pandas as pd 

from sklearn.model_selection import train_test_split

def load_pickle(name):
    return pd.read_pickle(f'../data/{name}.pkl')

def create_Xy(df):
    y = df.buy_event
    X = df.drop(columns='buy_event')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1234)

    return X, y, X_train, X_test, y_train, y_test

def run_models():
    obs = load_pickle('observations')

    # build X,y and perform train test split
    X, y, X_train, X_test, y_train, t_test = create_Xy(obs)

    return X_train

if __name__ == '__main__':
    print(run_models().info())