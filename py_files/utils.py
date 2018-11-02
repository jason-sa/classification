import pandas as pd 

def write_to_pickle(df,name):
        ''' Utility to write data frames to pickles
        '''
        df.to_pickle(f'../data/{name}.pkl')

def load_pickle(name):
        ''' Utility to load data frames from pickles
        '''
        return pd.read_pickle(f'../data/{name}.pkl')
