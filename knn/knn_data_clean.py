import pandas as pd

def load_data(file_path):
    # Load the data from CSV file
    data = pd.read_csv(file_path)

    # Drop the 'Datetime' column
    data = data.drop(labels=["Datetime"], axis=1)

    # Replace '#VALUE!' with 0
    data.replace('#VALUE!', '0', inplace=True)  # Convert '0' to a string

    # Convert the data to numeric (float) format
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
    #error=coerce converts the non-convertible value to nan


    # Fill missing values with the mean of each column
    data.fillna(data.mean(), inplace=True)#in this code mean of each column is calculated and add in the missing place of that valuein the respected columns

    # Rename columns as needed
    data.rename(columns={'AHU: SAT': 'AHU:Supply Air Temperature',
                         'AHU: MAT': 'AHU:Mixed Air Temperature',
                         'AHU: RAT': 'AHU:Return Air Temperature',

                         }, inplace=True)

    return data
