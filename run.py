from linearRegress_utils import *
from plot_utils import *
from utils import *

data_path = 'winequality-red.csv'
result_path = 'result.txt'

def run():
    # Load Data Set

    data = loadData(data_path)
    # print(data)
    # Make Train-Test Split
    train_data, train_X, train_y, test_X, test_y = splitData(data, 0.80)

    # Training Data Visualization
    visualizeData(train_data)

    # Trigger Linear Regression

    # Save Model

    # Save Results

    # Visualize results (Training Accuracy, Validation Accuracy and Testing Accuracy)
run()