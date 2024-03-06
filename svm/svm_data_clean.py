import pandas as pd

def clean_data(data):
    # Drop the 'Datetime' column
    data = data.drop(labels=["Datetime"], axis=1)

    # Replace '#VALUE!' with 0
    data.replace('#VALUE!', '0', inplace=True)  # Convert '0' to a string

    # Convert the data to numeric (float) format
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN

    # Fill missing values with the mean of each column
    data.fillna(data.mean(), inplace=True)  # Fill NaN values with the mean of each column

    # Rename columns as needed
    data.rename(columns={'AHU: SAT': 'AHU:Supply Air Temperature',
                         'AHU: MAT': 'AHU:Mixed Air Temperature',
                         'AHU: RAT': 'AHU:Return Air Temperature'}, inplace=True)

    return data
