''' Create feature set for modeling

Author: Jason Salazer-Adams
Date: 10/23/2018

'''
import pandas as pd
import numpy as np

# get transformed data
observations = pd.read_pickle('../data/observations.pkl')
prior_observations = pd.read_pickle('../data/prior_observations.pkl')

# days since last order and number times an item has been added to the cart
most_recent_order = (prior_observations[prior_observations.in_cart == 1]
                     .groupby(['visitorid','itemid'])['local_date_time']
                     .agg(['max', 'count'])
                     .reset_index())

observations = observations.merge(most_recent_order, on=['visitorid','itemid'], how='left')
observations['days_since_last_order'] = observations.local_date_time - observations['max']
observations['days_since_last_order'] = observations.days_since_last_order.dt.total_seconds()/60/60/24
observations.drop(columns='max', inplace=True)
observations.rename(columns={'count':'add_frequency'}, inplace=True)

# fill gaps, if never re-ordered in history then max days and 0 frequency
observations.days_since_last_order.fillna(observations.days_since_last_order.max(), inplace=True)
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