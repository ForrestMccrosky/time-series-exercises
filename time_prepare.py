import numpy as np
import seaborn as sns
import datetime
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

########################################## Prepare Function File #################################

def make_time(df):
    '''
    This function is set up to take in a dataframe with a date column and convert the date to date 
    time format and set it as an index as well of make the columns:

    month
    day_of_week
    sales_total
    '''

    ## making sale_date to datetime format
    df.sale_date = pd.to_datetime(df.sale_date)

    ## setting datetime column as the index
    df = df.set_index('sale_date').sort_index()
    
    ## making a month column
    df['month'] = df.index.month

    ## making a day of week column
    df['day_of_week'] = df.index.day_name()

    ## last but not least making a sales total column
    df['sales_total'] = df.sale_amount * df.item_price
    return df


def set_time(df):
    '''
    This function is designed to take in the german power dataframe and convert the 
    date column to datetime format and set it as an index

    as well as make month and year columns 
    '''
    
    ## making Date column datetime format
    df.Date = pd.to_datetime(df.Date)
    
    ## setting Date column as index
    df = df.set_index('Date')
    
    ## making month column
    df['month']=df.index.month
    
    ## making year column
    df['year']=df.index.year
    
    ## filling nulls with a 0
    df = df.fillna(0)

    ## returning dataframe
    return df



###################### Function To Split Data ###########################

def split_data(df):
    '''
    This function takes in a datframe and split it into the 
    train, validate, and test dataframes neccessary for proper modeling
    '''
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.3, random_state=123)

    return train, validate, test
