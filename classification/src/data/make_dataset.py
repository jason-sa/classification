#!/usr/bin/env python
# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import os
from datetime import datetime

from data_transformation import load, create_observations

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('transforming raw event data')

    events_file_path = os.path.join(input_filepath, 'events.csv')
    item_prop_1_file_path = os.path.join(input_filepath, 'item_properties_part1.csv')
    item_prop_2_file_path = os.path.join(input_filepath, 'item_properties_part2.csv')
    date_filter = datetime(2014, 9, 1)# filter the anlaysis to start on 9/1
    session_time_limit = 30 # new session will start after 30 mins of inactivity

    events_trimmed = load(  events_file_path,
                            item_prop_1_file_path,
                            item_prop_2_file_path,
                            date_filter,
                            session_time_limit)

    logger.info('creating the observation and prior observation data')
    observations, prior_observations = create_observations(events_trimmed, 2)

    logger.info(f'wrtiting files to {output_filepath}')

    events_out = os.path.join(output_filepath, 'events.csv')
    events_trimmed.to_csv(events_out, index=False)

    obs_out = os.path.join(output_filepath, 'observations.csv')
    observations.to_csv(obs_out, index=False)

    prior_out = os.path.join(output_filepath, 'prior_observations.csv')
    prior_observations.to_csv(prior_out, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # specify the input and output paths    
    input_path = os.path.join(project_dir, 'data', 'raw')
    output_path = os.path.join(project_dir, 'data', 'processed')

    main(input_path, output_path)
