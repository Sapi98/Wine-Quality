from linearRegress_utils import *
from plot_utils import *
from utils import *

data_path = 'winequality-red.csv'
result_path = 'result.txt'

def run():
    # Load Data Set

    data, X, y = loadData(data_path)
    # print(data)
    # Make Train-Test Split

    # Training Data Visualization
    visualizeData(data)

    # Trigger Linear Regression

    # Save Model

    # Save Results

    # Visualize results (Training Accuracy, Validation Accuracy and Testing Accuracy)
run()