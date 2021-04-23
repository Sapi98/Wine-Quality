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
    file_name = 'Plots/'

    if model.val_flag:
        file_name += str(model.max_iter) + '_' + str(model.alpha) + '_' + str(model.reg) + '_' + str(model.minibatch_size) + 'plot.png'

        train = pd.read_csv(model.record_cost)
        test = pd.read_csv(model.record_evaluation_testing)
        val = pd.read_csv(model.record_evaluation_validation)

        a1 = np.array(train['J'])[1:]
        a2 = np.array(test['MSE'])[1:]
        a3 = np.array(val['MSE'])[1:]

        x = np.arange(0, a1.shape[0])
        y = np.arange(0, a2.shape[0])
        z = np.arange(0, a3.shape[0])

        plt.plot(x, a1, label='Training Error')
        plt.plot(z, a3, label='Validation Error')
        plt.plot(y, a2, label='Testing Error')
            
    else:
        file_name += model.algo + '_' + str(model.max_iter) + '_' + str(model.alpha) + '_' + str(model.reg) 

        if model.algo == 'minibatch':
            file_name += '_' + str(model.minibatch_size) 
        
        file_name += 'plot.png'

        train = pd.read_csv(model.record_cost)
        test = pd.read_csv(model.record_evaluation_testing)

        #print(train)
        a1 = np.array(train['J'])[1:]
        a2 = np.array(test['MSE'])[1:]
        x = np.arange(0, a1.shape[0])
        y = np.arange(0, a2.shape[0])
        #print(a1)
        #print(test)

        plt.plot(x, a1, label='Training Error')
        plt.plot(y, a2, label='Testing Error')

    plt.xlabel('Iteration Number')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig(file_name, dpi=600)
    plt.clf()
    #plt.show()
    