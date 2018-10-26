''' Transform events raw data into sessions. Can also split the event data into an observation data frame
    and prior data frame

Author: Jason Salazer-Adams
Date: 10/24/2018

'''
import pandas as pd
import numpy as np
from datetime import datetime
import utils

pd.set_option('mode.chained_assignment',None)

DATE_FILTER = datetime(2015, 9, 1)
EVENTS_FILE = '../data/events.csv'
ITEM_PROP1 = '../data/item_properties_part1.csv'
ITEM_PROP2 = '../data/item_properties_part2.csv'
SESSION_TIME_LIMIT = 30 ## minutes between clicks

def convert_to_local(x):
    ''' Helper function to convert timestamp in milli-seconds to a date time
    '''
    return datetime.fromtimestamp(x/1000)

def calc_session_id(df, mask):
    ''' Calculates a web session id based on a data frame and a session filter logic

    df: data frame of events
    mask: filter on the data frame which defines a session

    return: pd.DataFrame (same as df with session_id added)
    '''

    df['session_id'] = np.nan
    ind = df.groupby('visitorid').head(1).index
    df.loc[ind, 'session_id'] = 1

    count_session = df[mask].shape[0] + 1
    df.loc[mask, 'session_id'] = np.arange(2,count_session+1)

    # fill in all of the gaps
    df.session_id.fillna(method = 'ffill', inplace=True)

    # make the session id unique
    df.session_id = df.visitorid.astype(str) + '_' + df.session_id.astype(int).astype(str)
    
    return df

def add_item_property(events_df, property_df, p_name):
    
    events_df.sort_values('local_date_time', inplace=True)
    property_df.sort_values('local_date_time', inplace=True)

    events_df = pd.merge_asof(events_df, property_df[property_df.property == p_name], 
                                   on='local_date_time',
                                   by='itemid')

    events_df = events_df.rename(columns={
                                                        'timestamp_x': 'timestamp',
                                                        'value': p_name
                                                    })

    events_df = events_df.drop(columns=['timestamp_y', 'property'])
    
    return events_df

def load():
        # get data
        events = pd.read_csv(EVENTS_FILE)
        item_p1 = pd.read_csv(ITEM_PROP1)
        item_p2 = pd.read_csv(ITEM_PROP2)

        events['local_date_time'] = events.timestamp.apply(convert_to_local)
        item_p1['local_date_time'] = item_p1.timestamp.apply(convert_to_local)
        item_p2['local_date_time'] = item_p2.timestamp.apply(convert_to_local)

        item_p = pd.concat([item_p1, item_p2], axis=0)
        
        print(f'Total number of events: {events.shape[0]:,}')
        print()

        # reduce data set size for MVP
        events_trimmed = events[events.local_date_time >= DATE_FILTER]

        print(f'Total number of trimmed events: {events_trimmed.shape[0]:,}')
        print()

        # calcualte the time diff within each session
        events_trimmed.sort_values(['visitorid', 'local_date_time'], inplace = True)

        # grouby + diff is slow may need to fix
        events_trimmed['minutes_since_prev_event'] = (events_trimmed
                                .groupby('visitorid')['timestamp']
                                .diff(1)
                                .fillna(0) 
                                / 1000
                                / 60)

        mask = (events_trimmed.minutes_since_prev_event > SESSION_TIME_LIMIT)
        events_trimmed = calc_session_id(events_trimmed, mask)

        events_trimmed['seq'] = (events_trimmed.session_id
                                                .str.split('_')
                                                .apply(lambda x: x[1])
                                                .astype(int))

        events_trimmed['seq'] = (events_trimmed
                                        .groupby('visitorid')['seq']
                                        .rank(method='dense'))

        events_trimmed = add_item_property(events_trimmed, item_p, 'categoryid')
        events_trimmed = add_item_property(events_trimmed, item_p, 'available')
        events_trimmed = add_item_property(events_trimmed, item_p, '790')
        events_trimmed = events_trimmed.rename(columns = {'790':'price'})

        events_trimmed['available'] = events_trimmed['available'].astype(float)
        
        return events_trimmed

def create_observations(df, seq):
        prior_observations = df[df.seq <= seq] # could get session != seq. i think could be OK or drop na in the feature creation
        prior_observations['buy_event'] = 0
        prior_observations.loc[prior_observations.event == 'transaction','buy_event'] = 1
        prior_observations = prior_observations.groupby(['session_id','seq'])['buy_event'].max().reset_index()

        prior_observations['visitor_id'] = (prior_observations.session_id
                                                .str.split('_')
                                                .apply(lambda x: x[0])
                                                .astype(int))
        

        observations = prior_observations[prior_observations.seq == seq]
        prior_observations = prior_observations[prior_observations.seq < seq]

        return observations, prior_observations

if __name__ == '__main__':
        events_trimmed = load()
        observations, prior_observations = create_observations(events_trimmed, 2)
        print(events_trimmed.head())
        utils.write_to_pickle(events_trimmed,'events_trimmed')
        utils.write_to_pickle(observations, 'observations')
        utils.write_to_pickle(prior_observations, 'prior_observations')

        '''
        # first calcualte sessions for each buy transaction
        events_trimmed.sort_values(['visitorid','local_date_time'], inplace=True)
        events_trimmed['prev_event'] = events_trimmed.groupby('visitorid').event.shift(1)
        sub = (events_trimmed.event == 'view') & (events_trimmed.prev_event == 'transaction')
        events_trimmed = calc_session_id(events_trimmed, sub)

        print(f'Total number of sessions dividing on transaction: {len(events_trimmed.session_id.unique()):,}')
        print()

        # calcualte the time diff within each session
        events_trimmed.sort_values(['session_id', 'local_date_time'], inplace = True)

        # grouby + diff is slow may need to fix
        events_trimmed['time_diff'] = (events_trimmed
                                .groupby('session_id')['timestamp']
                                .diff(1)
                                .fillna(0) 
                                / 1000)

        events_trimmed['page_length'] = events_trimmed.groupby('visitorid').time_diff.shift(-1)

        # re-calaculate sessions with a new session starting whenever a buy occurs or if a view lasts longer than 3.5 minutes
        sub = (((events_trimmed.event == 'view') 
        & (events_trimmed.prev_event == 'transaction'))
        | ((events_trimmed.event.isin(['view','addtocart']) )
                & (events_trimmed.time_diff > SESSION_TIME_LIMIT)))

        events_trimmed = calc_session_id(events_trimmed, sub)

        print(f'Total number of sessions adding a time limit: {len(events_trimmed.session_id.unique()):,}')
        print()

        # trim down to addtocart events only
        events_addtocart = events_trimmed[events_trimmed.event == 'addtocart']

        order_history = events_addtocart.loc[:,['session_id', 'visitorid','itemid', 'local_date_time']]

        # visitors with multiple sessions only '_1' is alwasy the first session for each visitor
        visitors = order_history[~order_history.session_id.str.endswith('_1')].visitorid.unique()
        order_history = order_history[order_history.visitorid.isin(visitors)]
        order_history = order_history.groupby(['session_id','visitorid','itemid'])['local_date_time'].max().reset_index()

        # remove dup items which were added twice in the event log
        order_history = order_history.groupby(['session_id','visitorid','itemid'])['local_date_time'].max().reset_index()

        unique_visitor_item = order_history.loc[:,['visitorid', 'itemid']].drop_duplicates()
        total_order_history = unique_visitor_item.merge(order_history, how='outer', on='visitorid')

        # Calaculate what was in the cart over time based on what the visitor has ever purchased
        total_order_history['in_cart'] = 0

        mask = (total_order_history.itemid_y == total_order_history.itemid_x)
        total_order_history.loc[mask, 'in_cart'] = 1

        # Clean-up columns
        total_order_history.drop(columns='itemid_y', inplace=True)
        total_order_history.rename(columns={'itemid_x':'itemid'}, inplace=True)

        # remove dups where the items are not in the cart (in_cart = 0)
        total_order_history['item_session'] = total_order_history.itemid.astype(str) + '_' + total_order_history.session_id
        total_order_history[total_order_history.in_cart == 1].item_session.unique()

        mask = ((total_order_history.item_session
                .isin(total_order_history[total_order_history.in_cart == 1]
                        .item_session
                        .unique()
                        ))
        & (total_order_history.in_cart == 0))
        total_order_history = total_order_history[~mask]
        total_order_history = (total_order_history
                        .groupby(['visitorid','itemid','session_id','in_cart','item_session'])['local_date_time']
                        .max()
                        .reset_index())

        # calacualte order sequence
        total_order_history['seq'] = (total_order_history.session_id
                                                .str.split('_')
                                                .apply(lambda x: x[1])
                                                .astype(int))

        total_order_history['order_seq'] = (total_order_history
                                        .groupby('visitorid')['seq']
                                        .rank(method='dense'))

        total_order_history.drop(columns='seq',inplace=True)

        # split last_order and all other orders
        last_order = total_order_history.groupby('visitorid')['session_id'].last()

        observations = total_order_history[total_order_history.session_id.isin(last_order)]
        prior_observations = total_order_history.drop(observations.index)

        # save data
        events_trimmed.to_pickle('../data/events_trimmed.pkl')
        total_order_history.to_pickle('../data/total_order_history.pkl')
        observations.to_pickle('../data/observations.pkl')
        prior_observations.to_pickle('../data/prior_observations.pkl')
        '''