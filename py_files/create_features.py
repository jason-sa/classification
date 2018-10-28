''' Add features to the observations data set for modeling

Author: Jason Salazer-Adams
Date: 10/24/2018

'''
import pandas as pd
# import numpy as np
import data_transformation as dt
import utils

def calc_view_counts(prior_df, events, c_name):
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df[merge_df.event == 'view']
    merge_df = merge_df.groupby(['visitor_id'])['event'].count().reset_index()
    merge_df = merge_df.rename(columns={'event':c_name})

    return merge_df

def calc_session_length(prior_df, events, c_name):
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df.groupby(['visitor_id'])['minutes_since_prev_event'].sum().reset_index()
    merge_df = merge_df.rename(columns={'minutes_since_prev_event':c_name})

    return merge_df

def calc_item_view_counts(prior_df, events, c_name):
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df[merge_df.event == 'view']
    merge_df = merge_df.groupby(['visitor_id'])['itemid'].nunique().reset_index()
    merge_df = merge_df.rename(columns={'itemid':c_name})

    return merge_df

def calc_add_counts(prior_df, events, c_name):
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df[merge_df.event == 'addtocart']
    merge_df = merge_df.groupby(['visitor_id'])['event'].count().reset_index()
    merge_df = merge_df.rename(columns={'event':c_name})

    return merge_df

def calc_transaction_counts(prior_df, events, c_name):
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df[merge_df.event == 'transaction']
    merge_df = merge_df.groupby(['visitor_id'])['event'].count().reset_index()
    merge_df = merge_df.rename(columns={'event':c_name})

    return merge_df

def calc_avg_avail_views(prior_df, events, c_name):
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df[merge_df.event == 'view']
    merge_df = merge_df.groupby(['visitor_id'])['available'].agg(['count','sum'])
    merge_df[c_name] = merge_df['sum'] / merge_df['count']
    merge_df = merge_df.drop(columns=['count','sum'])

    return merge_df

def calc_avg_price(prior_df, events, c_name):
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = merge_df.groupby(['visitor_id', 'itemid'])['price'].mean().reset_index()
    merge_df = merge_df.rename(columns={'price':c_name})

    return merge_df

def add_feature(obs, feature, c_name, na_val, on_cols = ['visitor_id']):
    obs = pd.merge(obs, feature, how='left', on=on_cols)
    if na_val is not None:
        obs.loc[:, c_name] = obs.loc[:, c_name].fillna(na_val)
    else:
        obs.loc[:, c_name] = obs.loc[:, c_name].dropna()

    return obs

def gen_features(events, prior_df, observations):
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
    

'''

# get transformed data
observations = pd.read_pickle('../data/observations.pkl')
prior_observations = pd.read_pickle('../data/prior_observations.pkl')

# days since last order and number times an item has been added to the cart
most_recent_order = (prior_observations[prior_observations.in_cart == 1]
                     .groupby(['visitorid','itemid'])['local_date_time']
                     .agg(['max', 'count'])
                     .reset_index())

observations = observations.merge(most_recent_order, on=['visitorid','itemid'], how='left')
# observations['days_since_last_order'] = observations.local_date_time - observations['max']
# observations['days_since_last_order'] = observations.days_since_last_order.dt.total_seconds()/60/60/24
observations.drop(columns='max', inplace=True)
observations.rename(columns={'count':'add_frequency'}, inplace=True)

# fill gaps, if never re-ordered in history then max days and 0 frequency
# observations.days_since_last_order.fillna(observations.days_since_last_order.max(), inplace=True)
observations.add_frequency.fillna(0, inplace=True)

# Average length per add to cart
prior_adds = prior_observations[prior_observations.in_cart == 1].reset_index()
prior_adds['prev_add'] = prior_adds.groupby(['visitorid','itemid'])['local_date_time'].shift(1)
prior_adds['days_prev_add'] = (prior_adds.local_date_time - prior_adds.prev_add).dt.total_seconds()/60/60/24
prior_adds = prior_adds.groupby(['visitorid', 'itemid'])['days_prev_add'].mean().reset_index()

observations = observations.merge(prior_adds, on=['visitorid', 'itemid'], how='left')
observations.days_prev_add.fillna(observations.days_prev_add.max(), inplace=True)

# create X,y
y = observations.in_cart
X = observations.drop(columns={'in_cart','visitorid','itemid','local_date_time','session_id','item_session','order_seq'})

y.to_pickle('../data/y.pkl')
X.to_pickle('../data/X.pkl')
observations.to_pickle('../data/observations_trans.pkl')
'''