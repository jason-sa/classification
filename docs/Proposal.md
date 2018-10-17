# Project 3 (McNulty) Proposal

Jason Salazer-Adams

## Overview

A general e-commerce site would like to answer a fundamental question of, "Can they predict a customer's implicit filters, e.g. product type, price range, etc.?" I am planning to attempt to address this question by utilziing a [Kaggle data set](https://www.kaggle.com/retailrocket/ecommerce-dataset/home) provided by [RetailRocket](https://retailrocket.net/). The data set was collected from a real e-commerce website and represents visitors buying behaviors in the form of view/addtocart/transaction.

* view - Visitor clicked on an item's page.
* addtocart - Visitor added the item to their shopping cart.
* transaction - Visitor bought the item.

There are three main data sets, 

1. Category tree - hierarchical data set relating the lowest level category to the top level parent
2. Events - The visitor's events on the e-commerce site
3. Item properties - Weekly recording of properties for an item, e.g. category, price, availability, etc.

The data has been anonymized and all properties have been hashed, except for category and avaialable. The value of the properties are also hashed, except for numerical data. Based on a response from retail rocket on Kaggle, it seem price and discount could be reveresed engineer. However, all other categories are not interpretable, which is a concern.

The purpose of the model is to be able to predict a property of the product which will be added to the visitor's cart. There will be bias in the model as I will only be looking at visitor's who eventually added an item to their cart, and there are visitors who do not add an item to their cart.

## Data Characteristics

I performed some basic EDA of the data sets as I was concerned about the "raw" internet traffic data. Here are some basic characteristics of the data.

## Features

## Challenges

## MVP