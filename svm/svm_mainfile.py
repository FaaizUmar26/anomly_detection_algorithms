import pandas as pd
import svm_data_loader
import svm_data_clean
import svm_model_trainer
import svm_model_accuracy
import svm_confusion_matrix
import svm_accuracy_chart
from sklearn.model_selection import train_test_split
import signal_plotter
from svm_fft import perform_fft
# Load your CSV file into a DataFrame
df = pd.read_csv('C:\pycharm\szcav_copy.csv')

# Convert the 'Datetime' column to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Input start and end time
start_time = pd.to_datetime(input("Enter the start time (YYYY-MM-DD HH:MM:SS): "))
end_time = pd.to_datetime(input("Enter the end time (YYYY-MM-DD HH:MM:SS): "))

# Call the function to plot signals in subplots with line graphs and rotated x-axis labels
signal_plotter.plot_signals_subplots(df, start_time, end_time)

file_path = 'C:\pycharm\szcav_copy.csv'

data = svm_data_loader.load_data(file_path)
data = svm_data_clean.clean_data(data)

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2, random_state=42)
classifier = svm_model_trainer.train_svm_classifier(X_train, y_train)
accuracy = svm_model_accuracy.test_model(classifier, X_test, y_test)

print(f'Accuracy: {accuracy}')

# Plot confusion matrix
classes = ['Fault', 'No Fault']  # Replace with your class names
svm_confusion_matrix.plot_confusion_matrix(y_test, classifier.predict(X_test), classes)
svm_accuracy_chart.visualize_accuracy(accuracy)

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
