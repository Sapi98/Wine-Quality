import matplotlib.pyplot as plt
import seaborn as sb

# pair plot using seaborn
# scatter plot heatmap
# contour plot
# line plot for performance

def visualizeData(data):
    for i in range(data.shape[1]-1 // 4):
        var = data.columns[i*4:(i+1)*4]
        sb.pairplot(data, vars=var, hue='quality', diag_kind='hist')
        plt.savefig('data_'+str(i)+'.png')
    
    var = data.columns[(data.shape[1]-1 // 4)*4:]
    sb.pairplot(data, vars=var, hue='quality', diag_kind='hist')
    plt.savefig('data_last.png')
    