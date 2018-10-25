import pandas as pd 

def write_to_pickle(df,name):
        df.to_pickle(f'../data/{name}.pkl')

def load_pickle(name):
    return pd.read_pickle(f'../data/{name}.pkl')
