from linearRegress_utils import *
from plot_utils import *
from utils import *

data_path = 'winequality-red.csv'
max_iter = 1000
alpha = [0.01, 0.03, 0.05, 0.07, 0.1]
#alpha = 0.1
reg = [0, 0.01, 0.05, 0.2, 0.3]
#reg = 0
val_flag = True
algo='batch'
models = []


def run( max_iter=100, alpha=0.01, reg=0, minibatch_size = 16, val_flag=True, algo='batch'):
    global models
    model = None

    # Load Data Set
    data = loadData(data_path)
    data_array = np.array(data)[1:]
    print('Loading Data Done')

    #Feature Normalization
    data_array[:,:-1] = featureNormalization(data_array[:,:-1])
    #print(data_array)
    
    # Make Train-Test Split
    train_data, train_X, train_y, test_X, test_y = splitData(data_array[:,:-1], data_array[:,-1], 0.80)
    print('Data Spliting Done')

    # Training Data Visualization
    visualizeData(train_data)
    print('Visualize Data Done')

    # Trigger Linear Regression
    print('Linear Regression Triggered')
    if val_flag:
        print('Mode : Validation')
        for a in alpha:
            for r in reg:
                #for s in minibatch_size:
                print('='*60)
                print('Alpha :', a, 'Regularization parameter Lambda :', r)
                print('='*60)
                train_X, train_y, val_X, val_y = generateValidation(train_X, train_y)
                model  = Model( max_iter, a, r, val_flag)
                model.fit(train_X, train_y, test_X, test_y, val_X, val_y, algo=algo)

                models.append(model)
                print('='*60)
    
    else:
        model  = Model(max_iter, alpha, reg, val_flag=val_flag)
        print('Model is built')
        print('Fitting model triggered')
        model.fit(train_X, train_y, test_X, test_y, algo=algo)
        print('Training model Done')
        models = [model]

    print('Training Model Done')

    # Save Model
    for model in models:
        model.save_weight()
    print('Weights Saved')
    

    # Visualize results (Training Accuracy, Validation Accuracy and Testing Accuracy)
    for model in models:
        visualizePerformance(model)
    print('Visualization Done')
    
run(max_iter, alpha, reg, val_flag=val_flag, algo=algo)