import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

palette = sb.color_palette("magma")

# pair plot using seaborn
# scatter plot heatmap
# contour plot
# line plot for performance

def visualizeData(data):
    for i in range((data.shape[1]-1) // 4):
        var = data.columns[i*4:(i+1)*4]
        sb.pairplot(data, vars=var, palette=palette, hue='quality', diag_kind='kde')
        plt.savefig('data_'+str(i)+'.png')
    
    var = data.columns[((data.shape[1]-1) // 4)*4:-1]
    sb.pairplot(data, vars=var, palette=palette, hue='quality', diag_kind='kde')
    plt.savefig('data_last.png')
    

def visualizePerformance(model):
    file_name = ''

    if model.val_flag:
        file_name = str(model.max_iter) + '_' + str(model.alpha) + '_' + str(model.reg) + '_' + str(model.minibatch_size) + 'plot.png'

        train = pd.read_csv(model.record_cost)
        test = pd.read_csv(model.record_evaluation_testing)
        val = pd.read_csv(model.record_evaluation_validation)

        x = train.shape()[0]

        plt.plot(x, train['MSE'], label='Training Error')
        plt.plot(x, val['MSE'], label='Validation Error')
        plt.plot(x, test['MSE'], label='Testing Error')
        
    else:
        file_name = 'realtime_' + str(model.max_iter) + '_' + str(model.alpha) + '_' + str(model.reg) + '_' + str(model.minibatch_size) + 'plot.png'

        train = pd.read_csv('realtime_' + model.record_cost)
        test = pd.read_csv('realtime_' + model.record_evaluation_testing)

        x = train.shape()[0]

        plt.plot(x, train['MSE'], label='Training Error')
        plt.plot(x, test['MSE'], label='Testing Error')

    plt.xlabel('Iteration Number')
    plt.ylabel('Error')
    plt.legend()

    plt.show()
    plt.savefig(file_name)