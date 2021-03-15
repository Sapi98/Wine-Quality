from linearRegress_utils import *
from plot_utils import *
from utils import *

data_path = 'winequality-red.csv'
result_path = 'result.txt'
max_iter = 1000
alpha = [0.01, 0.03, 0.05, 0.1]
reg = [0, 0.1, 0.3, 0.5, 0.7]
models = []

def run( max_iter=100, alpha=0.01, reg=0, minibatch_size = 16, val_flag=True):
    global models

    # Load Data Set

    data = loadData(data_path)
    
    # Make Train-Test Split
    train_data, train_X, train_y, test_X, test_y = splitData(data, 0.80)

    # Training Data Visualization
    visualizeData(train_data)

    # Trigger Linear Regression
    
    if val_flag:
        for a in alpha:
            for r in reg:
                for s in minibatch_size:
                    train_X, train_y, val_X, val_y = generateValidation(train_X, train_y)
                    model  = Model( max_iter, a, r, s, val_flag)
                    model.fit(train_X, train_y, test_X, test_y, val_X, val_y)

                    models.append(model)

    # Save Model
    model.save_weight()
    
    # Save Results

    # Visualize results (Training Accuracy, Validation Accuracy and Testing Accuracy)
    for model in models:
        visualizePerformance(model)

run()