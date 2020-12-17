# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    '''
    Input: 
        messages_filepath - str - path to the messages file
        categories_filepath - str - path to the categories file
    Output:
        df - dataframe - merged message and categoreis
    '''
    messages = pd.read_csv(messages_filepath).drop_duplicates()
    categories = pd.read_csv(categories_filepath).drop_duplicates()

    # merge datasets
    df = messages.merge(categories, on = 'id', how = 'left')
    return df



def clean_data(df):
    '''
    Input: df - dataframe
    Output: df - dataframe - cleaned df
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = list(row)

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        # change column to dummy
        categories[column] = np.where(categories[column]==0, 0, 1)
    
    # clean column names
    col_names = [x[:-2] for x in categories.columns]
    categories.columns = col_names

    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1).fillna(0)

    # drop duplicates
    df = df.drop_duplicates()

    return df



def save_data(df, database_filename):
    '''
    Input: df - dataframe
           database_filename - string - path for the database
    Output: None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists = 'replace')
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()