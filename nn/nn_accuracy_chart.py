import matplotlib.pyplot as plt

def plot_accuracy_chart(accuracy):
    # Plotting the accuracy chart
    plt.figure(figsize=(6, 4))
    plt.bar(['Accuracy'], [accuracy], color='skyblue')
    plt.title('Model Accuracy')
    plt.xlabel('Metric')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis limits between 0 and 1
    plt.show()
