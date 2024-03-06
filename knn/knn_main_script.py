import pandas as pd #pandas library is used for data analysis used in machine learning task
import knn_data_clean#importing the data cleaning file to the main file
import knn_data_loader#importing the data loader file
import knn_model_trainer#importing the model trianer file to the main file
import knn_model_accuracy#importing the model accuracy file to the main file
import knn_confusion_matrix#importing the confusion matrix file to the main file
import knn_acuuracy_graph#importing the accuracy graph file to the main file
import signal_plotter#importing signal plotter file to the main file.
from knn_fft import perform_fft

file_path = 'C:\pycharm\SZCAV.csv'
df = pd.read_csv(file_path)
# Convert the 'Datetime' column to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Input start and end time
start_time = pd.to_datetime(input("Enter the start time (YYYY-MM-DD HH:MM:SS): "))#user inputs the start time
end_time = pd.to_datetime(input("Enter the end time (YYYY-MM-DD HH:MM:SS): "))#user inputs the end time.

# Call the function to plot signals in subplots with line graphs and rotated x-axis labels
signal_plotter.plot_signals_subplots(df, start_time, end_time)

# Load and clean the data
data = knn_data_clean.load_data(file_path)

# Split the data into training and testing datasets
X_train, X_test, y_train, y_test = knn_data_loader.split_data(data)

# Train the KNN classifier
classifier = knn_model_trainer.train_knn_classifier(X_train, y_train)

# Test the model's accuracy
accuracy = knn_model_accuracy.test_model(classifier, X_test, y_test)
print(f'Accuracy: {accuracy}')

# Visualize accuracy
knn_acuuracy_graph.visualize_accuracy(accuracy)

# Plot confusion matrix
classes = ['Fault', 'No Fault']  # Replace with your class names
y_pred = classifier.predict(X_test)
knn_confusion_matrix.plot_confusion_matrix(y_test, y_pred, classes)

# Apply FFT to specified columns
columns_to_fft = [
    "AHU: Supply Air Temperature", "AHU: Supply Air Temperature Heating Set Point",
    "AHU: Supply Air Temperature Cooling Set Point", "AHU: Outdoor Air Temperature",
    "AHU: Return Air Temperature", "AHU: Supply Air Fan Status",
    "AHU: Supply Air Fan Speed Control Signal", "AHU: Return Air Damper Control Signal",
    "AHU: Cooling Coil Valve Control Signal", "AHU: Heating Coil Valve Control Signal",
    "Occupancy Mode Indicator"
]

perform_fft(data, columns_to_fft)