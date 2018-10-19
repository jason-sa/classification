''' Feature generation for baseline MVP (minimum viable product)

Author: Jason Salazer-Adams
Date: 10/19/2018

'''
import pandas as pd
import numpy as np
from datetime import datetime

pd.set_option('mode.chained_assignment',None)

DATE_FILTER = datetime(2015, 8, 15)
EVENTS_FILE = '../data/events.csv'
SESSION_TIME_LIMIT = 600 ## Should consider building a time limit per user, instead of global

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

def buy_event(df):
    ''' Calcualtes whether or not a session results in a buy

    df: pd.DataFrame of events

    return pd.Series (session_id, [0,1] no buy/buy)
    '''

    df['buy_event'] = 0
    df.loc[df.event == 'transaction', 'buy_event'] = 1

    return df.groupby('session_id')['buy_event'].max()

def view_counts(df):
    ''' Calcualtes the number of views in each session

    df: pd.DataFrame of events

    return pd.Series (session_id, count of views)
    '''

    view_counts_df = (df[df.event == 'view']
                     .groupby('session_id')['event']
                     .count())
    view_counts_df.name = 'view_count'

    return view_counts_df

def session_length(df):
    ''' Calacualtes a data frame summarizing total session length and average session length

    df: pd.DataFrame of events

    return pd.DataFrame (session_id, sum of session, avg of session)
    '''

    session_length_df = (df
                        .groupby('session_id')['page_length']
                        .agg(['sum','mean']))
    session_length_df.rename(columns = {'sum':'session_length','mean':'avg_len_per_pg'}, inplace=True)

    return session_length_df

# get data
events = pd.read_csv(EVENTS_FILE)
events['local_date_time'] = events.timestamp.apply(convert_to_local)
print(f'Total number of events: {events.shape[0]:,}')
print()

# reduce data set size for MVP
events_trimmed = events[events.local_date_time >= DATE_FILTER]

print(f'Total number of events: {events_trimmed.shape[0]:,}')
print()

# first calcualte sessions for each buy transaction
events_trimmed.sort_values(['visitorid','local_date_time'], inplace=True)
events_trimmed['prev_event'] = events_trimmed.groupby('visitorid').event.shift(1)
sub = (events_trimmed.event == 'view') & (events_trimmed.prev_event == 'transaction')
events_trimmed = calc_session_id(events_trimmed, sub)

print(f'Total number of sessions dividing on transaction: {len(events_trimmed.session_id.unique()):,}')
print()

# calcualte the time diff within each session
events_trimmed.sort_values(['session_id', 'local_date_time'], inplace = True)

# grouby + diff is slow
events_trimmed['time_diff'] = (events_trimmed
                               .groupby('session_id')['timestamp']
                               .diff(1)
                               .fillna(0) 
                               / 1000)

events_trimmed['page_length'] = events_trimmed.groupby('visitorid').time_diff.shift(-1)

# re-calaculate sessions with a new session starting whenever a buy occurs or if a view lasts longer than 3.5 minutes
sub = (((events_trimmed.event == 'view') 
       & (events_trimmed.prev_event == 'transaction'))
      | ((events_trimmed.event == 'view')
        & (events_trimmed.time_diff > SESSION_TIME_LIMIT)))

events_trimmed = calc_session_id(events_trimmed, sub)

print(f'Total number of sessions adding a time limit: {len(events_trimmed.session_id.unique()):,}')
print()

## Build the observation data set
feature_dfs = []

feature_dfs.append(buy_event(events_trimmed))
feature_dfs.append(view_counts(events_trimmed))
feature_dfs.append(session_length(events_trimmed))

design_df = pd.concat(feature_dfs,axis=1, sort=True)
design_df = design_df.fillna(0)
design_df.to_pickle('../data/design.pkl')

print(f'Design observations: {design_df.shape}')
print()

print(f'Precentage of buys: {len(design_df[design_df.buy_event >= 1].buy_event) / len(design_df.buy_event):.2%}')
