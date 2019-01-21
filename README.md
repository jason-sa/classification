# Will a customer buy or not?

## Summary

This repository addresses a classification problem in the e-commerce domain, and in particular attempting to predict whether or not a visitor to a site will buy or not upon their next visit.

The data utilized in this project was from Kaggle and provided by [RetailRocket](https://www.kaggle.com/retailrocket/ecommerce-dataset/home). The data were gathered from a real e-commerce website between May 2015 to September 2015. The data set consisted of 3 different types of data,

1. events - Transaction log of every event logged by a visitor.
2. item properties - Transaction log by week of each item property, i.e. categoryid or item availability.
3. category tree - Hierarchal data relating categories to parent categories. This data was not utilized in this analysis.

The data was highly anonymized to protect the identity of the e-commerce web site. The more interesting data, such as price, discount, item category names, etc., were all hidden. However, there was information about the behavior of each visitor. The behavior tracked in the data set were the following 3 events,

* view - A visitor navigated to a page to view an item.
* addtocart - A visitor decided to add the item they were viewing to their virtual cart.
* transaction - A visitor decided to buy the item that was added to their virtual cart.

Given this rich information about each visitor, then I wanted to try and predict whether or not a visitor will buy any item, based solely on how the visitor interacted with the site previously. Here are three reasons why an e-commerce business would be interested in predicting if a visitor will predict the next time they visit their site.

1. Knowing who will buy next can result in targeted marketing to those who will not buy.
2. Implementing a targeted marketing strategy results in an optimal allocation of a marketing budget.
3. Profit erosion will be minimized as all discounts or promotional programs are targeted to the right visitors.

## Repository Structure

* [Initial Proposal](docs/Proposal.pdf) - Initial model proposal based on data search and EDA of the selected data set.
* [Presentation](docs/Buy_or_Not.pdf) - High-level presentation of the results.
* [Summary](https://jason-sa.github.io/2018-11-02-Will-you-buy-or-not-in-your-next-visit/) - Blog post of the results.

# Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for anyone using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Presentations and proposal from original project.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         and a short description.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── train_model.py
    │   │

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
