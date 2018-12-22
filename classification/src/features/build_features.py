#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Add features to the observations data set for modeling

Author: Jason Salazer-Adams
Date: 10/24/2018
Updated: 12/22/18

'''
import pandas as pd
import os
from pathlib import Path
import logging


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
    merge_df = merge_df.rename(columns={'event': c_name})

    return merge_df


def calc_session_length(prior_df, events, c_name):
    ''' Calculates the length of the session

    prior_df: data frame defining the data prior to the observation
    events: data frame of all events
    c_name: name to give the feature

    return: pd.DataFrame of prior + c_name
    '''
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = (merge_df.groupby(['visitor_id'])['minutes_since_prev_event']
                        .sum()
                        .reset_index())
    merge_df = merge_df.rename(columns={'minutes_since_prev_event': c_name})

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
    merge_df = (merge_df.groupby(['visitor_id'])['itemid']
                        .nunique()
                        .reset_index())
    merge_df = merge_df.rename(columns={'itemid': c_name})

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
    merge_df = merge_df.rename(columns={'event': c_name})

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
    merge_df = merge_df.rename(columns={'event': c_name})

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
    merge_df = (merge_df.groupby(['visitor_id'])['available']
                        .agg(['count', 'sum']))
    merge_df[c_name] = merge_df['sum'] / merge_df['count']
    merge_df = merge_df.drop(columns=['count', 'sum'])

    return merge_df


def calc_avg_price(prior_df, events, c_name):
    ''' Calculates the averge price

    prior_df: data frame defining the data prior to the observation
    events: data frame of all events
    c_name: name to give the feature

    return: pd.DataFrame of prior + c_name
    '''
    merge_df = pd.merge(prior_df, events, on='session_id')
    merge_df = (merge_df.groupby(['visitor_id', 'itemid'])['price']
                        .mean()
                        .reset_index())
    merge_df = merge_df.rename(columns={'price': c_name})

    return merge_df


def add_feature(obs, feature, c_name, na_val, on_cols=['visitor_id']):
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


def gen_features():
    '''
    Main function to generate all features. The data are read from
    /data/processed

    retrun: csv
    Creates csv in /data/processed with the features added
    '''
    logger = logging.getLogger(__name__)

    #  load the processed data
    logger.info('Loading data')
    project_dir = Path(__file__).resolve().parents[2]
    processed_path = os.path.join(project_dir, 'data', 'processed')

    events = pd.read_csv(os.path.join(processed_path, 'events.csv'))
    prior_df = (pd.read_csv(
                os.path.join(processed_path, 'prior_observations.csv')))
    observations = (pd.read_csv(
                os.path.join(processed_path, 'observations.csv')))

    # calculate features
    logger.info('Calculating view counts.')
    view_counts = calc_view_counts(prior_df, events, 'view_count')

    logger.info('Calculating session length.')
    session_length = calc_session_length(prior_df, events, 'session_length')

    logger.info('Calculating item views.')
    item_views = calc_item_view_counts(prior_df, events, 'item_views')

    logger.info('Calculating addtocart events.')
    addtocart_counts = calc_add_counts(prior_df, events, 'add_to_cart_count')

    logger.info('Calculating transaction events.')
    transaction_counts = (calc_transaction_counts(prior_df,
                                                  events,
                                                  'transaction_count'))

    logger.info('Calculating average availability.')
    average_item_avail = calc_avg_avail_views(prior_df, events, 'avg_avail')
    # item_price = calc_avg_price(prior_df, events, 'avg_price')

    # add features to observations
    logger.info('Adding features to the observation data.')
    observations = (add_feature(observations,
                                view_counts,
                                'view_count',
                                0))
    observations = (add_feature(observations,
                                session_length,
                                'session_length',
                                -1))
    observations = (add_feature(observations,
                                item_views,
                                'item_views',
                                0))
    observations = (add_feature(observations,
                                addtocart_counts,
                                'add_to_cart_count',
                                0))
    observations = (add_feature(observations,
                                transaction_counts,
                                'transaction_count',
                                0))
    observations = (add_feature(observations,
                                average_item_avail,
                                'avg_avail',
                                None))

    # number of different categories viewed
    # availability of the items
    # maybe last category and then figure out how to one-hot encode

    logger.info('Writing observations_features.')
    observations = observations.dropna()
    (observations.to_csv(
                    os.path.join(processed_path, 'observations_features.csv'),
                    index=False))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    gen_features()
