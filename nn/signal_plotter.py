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
    num_rows = int(num_subplots ** 0.5) + 1#calculate square root total number of plots round off to nearest whole
    #number and +1 is for space for all subplots.
    num_cols = int(num_subplots / num_rows) + 1

    # Create a figure and axis objects for subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
#fig is the whole plot and axs is an array where each element represent an individual subplot.
    # Flatten the axis array to easily iterate over it
    axs = axs.flatten()

    # Plot each signal in a subplot with a line graph
    for i, signal in enumerate(signals):#enumerate is used to get both the index and value of each signal present in array
        ax = axs[i]#selecting a specific subplot
        ax.plot(filtered_df['Datetime'], filtered_df[signal], marker='o', linestyle='-')
        ax.set_title(signal)
        ax.set_xlabel('Datetime')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels

    # Hide any unused subplots
    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axs[i])

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()

    # Show the plot
    plt.show()
