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

DATE_FILTER = datetime(2014, 9, 1)
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
    ''' Adds item properties to the events data frame

    events_df: events data frame
    property_df: property data frame
    p_name: property value to add to the events data frame

    returns: pd.DataFrame (events + p_name)
    '''
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
        ''' Loads all raw data from RetailRocket, calculates session, and adds item properties

        returns: pd.DataFrame (events)
        '''
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

        if events.shape[0] != events_trimmed.shape[0]:
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
        
        print('Session calcualted and sequenced.')
        print()

        events_trimmed = add_item_property(events_trimmed, item_p, 'categoryid')
        events_trimmed = add_item_property(events_trimmed, item_p, 'available')
        # events_trimmed = add_item_property(events_trimmed, item_p, '790')

        print('Added category and item availability property.')
        events_trimmed['available'] = events_trimmed['available'].astype(float)

        # events_trimmed = events_trimmed.rename(columns = {'790':'price'}) # need to strip n and convert to float
        # events_trimmed['price'] = (events_trimmed['price']
        #                                 .str.replace('n','')
        #                                 .astype(float))
        return events_trimmed

def create_observations(df, seq):
        ''' Creates the observations and prior observations data frame

        df: events data frame
        seq: filter of how many sessions to filter for the analysis. seq = 2, means seq 1 is prior and seq 2 is observations

        returns (pd.DataFrame, pd.DataFrame) (observations, prior)
        '''
        visitors = df[df.seq == seq].visitorid.unique()
        prior_observations = df[(df.visitorid.isin(visitors)) & (df.seq <= seq)] 
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