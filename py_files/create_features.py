''' Add features to the observations data set for modeling

Author: Jason Salazer-Adams
Date: 10/24/2018

'''
import pandas as pd
# import numpy as np
import data_transformation as dt
import utils

def calc_view_counts(prior_df, events, c_name):
    ''' Counts the number of view events

    prior_df: data frame defining the data prior to the observation
    events: data frame of all events
    c_name: name to give the feature

    return: pd.DataFrame of prior + c_name
    '''
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df[merge_df.event == 'view']
    merge_df = merge_df.groupby(['visitor_id'])['event'].count().reset_index()
    merge_df = merge_df.rename(columns={'event':c_name})

    return merge_df

def calc_session_length(prior_df, events, c_name):
    ''' Calculates the length of the session

    prior_df: data frame defining the data prior to the observation
    events: data frame of all events
    c_name: name to give the feature

    return: pd.DataFrame of prior + c_name
    '''
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df.groupby(['visitor_id'])['minutes_since_prev_event'].sum().reset_index()
    merge_df = merge_df.rename(columns={'minutes_since_prev_event':c_name})

    return merge_df

def calc_item_view_counts(prior_df, events, c_name):
    ''' Counts the number of items viewed

    prior_df: data frame defining the data prior to the observation
    events: data frame of all events
    c_name: name to give the feature

    return: pd.DataFrame of prior + c_name
    '''
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df[merge_df.event == 'view']
    merge_df = merge_df.groupby(['visitor_id'])['itemid'].nunique().reset_index()
    merge_df = merge_df.rename(columns={'itemid':c_name})

    return merge_df

def calc_add_counts(prior_df, events, c_name):
    ''' Counts the number of add to cart events

    prior_df: data frame defining the data prior to the observation
    events: data frame of all events
    c_name: name to give the feature

    return: pd.DataFrame of prior + c_name
    '''
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df[merge_df.event == 'addtocart']
    merge_df = merge_df.groupby(['visitor_id'])['event'].count().reset_index()
    merge_df = merge_df.rename(columns={'event':c_name})

    return merge_df

def calc_transaction_counts(prior_df, events, c_name):
    ''' Counts the number of transaction events

    prior_df: data frame defining the data prior to the observation
    events: data frame of all events
    c_name: name to give the feature

    return: pd.DataFrame of prior + c_name
    '''
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df[merge_df.event == 'transaction']
    merge_df = merge_df.groupby(['visitor_id'])['event'].count().reset_index()
    merge_df = merge_df.rename(columns={'event':c_name})

    return merge_df

def calc_avg_avail_views(prior_df, events, c_name):
    ''' Calculates the average item availability of the session

    prior_df: data frame defining the data prior to the observation
    events: data frame of all events
    c_name: name to give the feature

    return: pd.DataFrame of prior + c_name
    '''
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df[merge_df.event == 'view']
    merge_df = merge_df.groupby(['visitor_id'])['available'].agg(['count','sum'])
    merge_df[c_name] = merge_df['sum'] / merge_df['count']
    merge_df = merge_df.drop(columns=['count','sum'])

    return merge_df

def calc_avg_price(prior_df, events, c_name):
    ''' Calculates the averge price

    prior_df: data frame defining the data prior to the observation
    events: data frame of all events
    c_name: name to give the feature

    return: pd.DataFrame of prior + c_name
    '''
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df.groupby(['visitor_id', 'itemid'])['price'].mean().reset_index()
    merge_df = merge_df.rename(columns={'price':c_name})

    return merge_df

def add_feature(obs, feature, c_name, na_val, on_cols = ['visitor_id']):
    ''' Adds the features to the observation data frame

    obs: data frame of observations
    feature: feature to be added to the observation data frame
    c_name: name of the features
    na_val: how to fill in NAs
    on_cols: how to merge observations and features

    return: pd.DataFrame of observations + c_name
    '''
    obs = pd.merge(obs, feature, how='left', on=on_cols)
    if na_val is not None:
        obs.loc[:, c_name] = obs.loc[:, c_name].fillna(na_val)
    else:
        obs.loc[:, c_name] = obs.loc[:, c_name].dropna()

    return obs

def gen_features(events, prior_df, observations):
    ''' Main function to generate all features

    events: data frame of events
    prior_df: data frame specifying the prior observations
    observations: data frame specifying the observations which will be predicted

    retrun: pd.DataFrame (observations with all features)
    '''
    # calculate features
    view_counts = calc_view_counts(prior_df, events, 'view_count')
    session_length = calc_session_length(prior_df, events, 'session_length')
    item_views = calc_item_view_counts(prior_df, events, 'item_views')
    addtocart_counts = calc_add_counts(prior_df, events, 'add_to_cart_count')
    transaction_counts = calc_transaction_counts(prior_df, events, 'transaction_count')
    average_item_avail = calc_avg_avail_views(prior_df, events, 'avg_avail')
    # item_price = calc_avg_price(prior_df, events, 'avg_price')

    # add features to observations (turn this into a loop with **kwargs if time permits)
    # features = {}
    observations = add_feature(observations, view_counts, 'view_count', 0)
    observations = add_feature(observations, session_length, 'session_length', -1)
    observations = add_feature(observations, item_views, 'item_views',0)
    observations = add_feature(observations, addtocart_counts, 'add_to_cart_count', 0)
    observations = add_feature(observations, transaction_counts, 'transaction_count', 0)
    observations = add_feature(observations, average_item_avail, 'avg_avail', None) # no history of item availabiity, then can't be modeled

    # number of different categories viewed
    # availability of the items
    # maybe last category and then figure out how to one-hot encode
    
    observations = observations.dropna()

    return observations

if __name__ == '__main__':
    prior_df = utils.load_pickle('prior_observations')
    events = utils.load_pickle('events_trimmed')
    observations = utils.load_pickle('observations')

    obs = gen_features(events, prior_df, observations)
    obs.head()
    utils.write_to_pickle(obs, 'features')
    print(obs.info())
    print(obs.describe())