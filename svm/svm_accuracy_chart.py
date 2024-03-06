import matplotlib.pyplot as plt

def visualize_accuracy(accuracy):

    plt.bar(['Accuracy'], [accuracy], color='blue')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis limits between 0 and 1
    plt.show()
