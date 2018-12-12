#!/usr/bin/env bash

# Retrieves the raw data from Kaggle
# Assume kaggle command line tool is installed and configured
kaggle datasets download -d retailrocket/ecommerce-dataset
unzip ecommerce-dataset.zip 
rm ecommerce-dataset.zip