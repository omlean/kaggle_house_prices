import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_clean_data(dataset, split=True):
    """Loads and cleans either training or test dataset into dataframe and cleans.
    Input parameters:
    dataset(str): 'train' or 'test'"""

    # load training or test dataset
    if dataset == 'train':
        df = pd.read_csv('data/train.csv')
    elif dataset == 'test':
        df = pd.read_csv('data/test.csv')
        
    # convert MSSubClass column to string (Categorical)
    df['MSSubClass'] = df['MSSubClass'].astype(str)

    # impute with most common value
    impute_mode_cols = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional']
    for col in impute_mode_cols:
        top_value = df[col].value_counts().index[0]
        df[col] = df[col].fillna(top_value)

    # impute with mean
    impute_mean_cols = ['LotFrontage', 'MasVnrArea']
    for col in impute_mean_cols:
        mean = df[col].mean()
        df[col] = df[col].fillna(mean)

    # impute with hand-chosen value
    impute_values = {
        'MasVnrType': 'None',
        'KitchenQual': 'TA',
        'GarageYrBlt': '0',
        'Electrical': 'SBrkr'
        }

    # null values for BsmtQual also have null-like values for other basement columns - assume no basement
    # Number fireplaces is 0 for every null row of FireplaceQu. Same for GarageType
    NAs = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for col in NAs:
        impute_values[col] = 'NA'

    zeros = ['BsmtFinSF1', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']
    for col in zeros:
        impute_values[col] = 0.0

    for col, value in impute_values.items():
        df[col] = df[col].fillna(value)

    # drop columns with mostly null values
    mostly_null_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
    df.drop(columns=mostly_null_cols, inplace=True)
    
    # create MM/YYYY column
    def date(row):
        yr = row.YrSold
        mo = row.MoSold
        date = datetime.date(year=yr, month=mo, day=1)
        return date.strftime('%Y-%m')

    df['sell_date'] = df.apply(date, axis=1)
    
    # if loading training dataset, split into training and validation set
    if dataset == 'train' and split:
        df_train, df_test = train_test_split(df, test_size=0.2)
        return df_train, df_test
    else:
        return df
    
def kaggle_score(y_true, y_pred):
    """Returns the RMSE between the log of the predicted sales price and the log of the actual sale price"""
    return mean_squared_error(np.log(y_true), np.log(y_pred), squared=False)

def make_pred_df(y_pred, filename):
    """Creates a properly formatted .csv file from the predicted values of the test set for submission to Kaggle.
    y_pred [array]: array of predicted values
    filename [str]: name of .csv file
    """
    df = load_clean_data('test').set_index('Id')
    assert len(y_pred) == len(df)
    df['SalePrice'] = y_pred
    df[['SalePrice']].to_csv(filename)