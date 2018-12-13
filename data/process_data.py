import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Loads ands merge message and categories csv files
    
    Note: The first line of each file must be the column names and
        both csv files must have an 'id' column.
        
    Args:
        message_filepath (str)
        categories_filepath (str)
    
    Returns:
        df (pandas.DataFrame)
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id', how='outer')
    return df


def clean_data(df):
    '''Cleans the Disaster Reponse Pipeline dataframe
        - Splits categories into separate columns with 0 or 1 as values
        - Drops 'child_alone' category
        - Drops duplicates

    Args:
        df (pandas.DataFrame)

    Returns:
        df (pandas.DataFrame): cleaned dataframe
    '''
    # Split categories into separate category columns
    categories = df.categories.str.split(';', expand=True)
    categories.columns = categories.loc[0].apply(lambda x: x[:-2])
    
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].apply(int)
    
    # Drop the child_alone category because all values are zeros 
    categories.drop('child_alone', axis='columns', inplace=True)
    
    df.drop('categories', axis='columns', inplace=True)
    df = pd.concat([df, categories], axis='columns')

    df.drop_duplicates(subset='id', inplace=True)
    return df


def save_data(df, database_filename):
    '''Saves the data as an SQLite database
    
    Args:
        df (pandas.DataFrame)
        data_filename (str)
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, if_exists='replace', index=False)
    return None

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