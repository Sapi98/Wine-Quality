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
    total_iter = np.arange(model.max_iter)

    train = pd.read_csv(model.record_cost)
    test = pd.read_csv(model.record_evaluation_testing)
    val = pd.read_csv(model.record_evaluation_validation)

    plt.plot(model.record_cost, total_iter, label='Training Cost')