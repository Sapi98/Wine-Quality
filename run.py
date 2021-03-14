from linearRegress_utils import *
from plot_utils import *
from utils import *

data_path = 'winequality-red.csv'
result_path = 'result.txt'

def run( max_iter=100, alpha=0.01, reg=0, minibatch_size = 16, val_flag=True):
    # Load Data Set

    data = loadData(data_path)
    
    # Make Train-Test Split
    train_data, train_X, train_y, test_X, test_y = splitData(data, 0.80)

    # Training Data Visualization
    visualizeData(train_data)

    # Trigger Linear Regression
    model  = Model( max_iter, alpha, reg, minibatch_size, val_flag)
    
    if val_flag:
        for a in alpha:
            for r in reg:
                for s in minibatch_size:
                    train_X, train_y, val_X, val_y = generateValidation(train_X, train_y)
                    model.fit()

    # Save Model
    model.save_weight()
    # Save Results

    # Visualize results (Training Accuracy, Validation Accuracy and Testing Accuracy)

run()