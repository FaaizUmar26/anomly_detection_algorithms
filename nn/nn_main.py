import nn_data_clean
import nn_model_accuracy
import nn_model_trainer
import nn_confusion_matrix
import nn_accuracy_chart
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import signal_plotter
from nn_fft import perform_fft

file_path = 'C:/pycharm/SZCAV.csv'
df = pd.read_csv(file_path)
# Convert the 'Datetime' column to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Input start and end time
start_time = pd.to_datetime(input("Enter the start time (YYYY-MM-DD HH:MM:SS): "))
end_time = pd.to_datetime(input("Enter the end time (YYYY-MM-DD HH:MM:SS): "))

# Call the function to plot signals in subplots with line graphs and rotated x-axis labels
signal_plotter.plot_signals_subplots(df, start_time, end_time)
# Load and clean data using nn_data_clean module
cleaned_df = nn_data_clean.load_data(file_path)

# Define feature columns and target column
feature_columns = [
    "AHU: Supply Air Temperature", "AHU: Supply Air Temperature Heating Set Point",
    "AHU: Supply Air Temperature Cooling Set Point", "AHU: Outdoor Air Temperature",
    "AHU: Return Air Temperature", "AHU: Supply Air Fan Status",
    "AHU: Supply Air Fan Speed Control Signal", "AHU: Return Air Damper Control Signal",
    "AHU: Cooling Coil Valve Control Signal", "AHU: Heating Coil Valve Control Signal",
    "Occupancy Mode Indicator"
]
target_column = "Fault Detection Ground Truth"

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(cleaned_df[feature_columns], cleaned_df[target_column], test_size=0.2, random_state=42)

# Build model
model = nn_model_trainer.build_model()

# Train model
history = nn_model_trainer.train_model(model, X_train, y_train)

# Get model accuracy
test_accuracy = nn_model_accuracy.get_model_accuracy(model, X_test, y_test)
print("Model Accuracy:", test_accuracy)

# Plot confusion matrix
# Get the predicted probabilities for each class
y_pred_prob = model.predict(X_test)

# Convert the probabilities to class labels by rounding to the nearest integer
y_pred = np.round(y_pred_prob).astype(int)

classes = ['Fault ', 'no fault']  # Replace with your class names
nn_confusion_matrix.plot_confusion_matrix(y_test, y_pred, classes)

nn_accuracy_chart.plot_accuracy_chart(test_accuracy)
columns_to_fft = [
    "AHU: Supply Air Temperature", "AHU: Supply Air Temperature Heating Set Point",
    "AHU: Supply Air Temperature Cooling Set Point", "AHU: Outdoor Air Temperature",
    "AHU: Return Air Temperature", "AHU: Supply Air Fan Status",
    "AHU: Supply Air Fan Speed Control Signal", "AHU: Return Air Damper Control Signal",
    "AHU: Cooling Coil Valve Control Signal", "AHU: Heating Coil Valve Control Signal",
    "Occupancy Mode Indicator"
]

# Call the function to perform FFT
perform_fft(df, columns_to_fft)
