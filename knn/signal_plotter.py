import pandas as pd
import matplotlib.pyplot as plt

def plot_signals_subplots(df, start_time, end_time):
    # Filter the DataFrame based on the start and end time
    filtered_df = df[(df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)]

    # Get the list of all columns except for 'Datetime'
    signals = filtered_df.columns[1:]

    # Calculate the number of subplots needed
    num_subplots = len(signals)

    # Calculate the number of rows and columns for subplots
    num_rows = int(num_subplots ** 0.5) + 1 #this calculate the square root of num_subplots.and add 1 to ensure
    # that there is one row and one column on it.one more purpose of adding one is if the square root is in decimal
    #that will convert into the nearest integer.

    num_cols = int(num_subplots / num_rows) + 1#calculate that how many columns are need in each row to fill plots

    # Create a figure and axis objects for subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Flatten the axis array to easily iterate over it
    axs = axs.flatten()

    # Plot each signal in a subplot with a line graph
    for i, signal in enumerate(signals):
        ax = axs[i]
        ax.plot(filtered_df['Datetime'], filtered_df[signal], marker='o', linestyle='-')
        ax.set_title(signal)#
        ax.set_xlabel('Datetime')#at x-axis this label and the time values are shown.
        ax.set_ylabel('Value')#at y-axis this label"value"is shown.
        ax.grid(True)# add grid lines for better readability.
        ax.tick_params(axis='x', rotation=45)  #rotate the x-axis at 45 degree.

    # Hide any unused subplots
    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs[i])#is a method that removes the selected subplot axis from the figure

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()

    # Show the plot
    plt.show()
