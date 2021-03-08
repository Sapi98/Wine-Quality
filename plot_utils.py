import matplotlib.pyplot as plt
import seaborn as sb

# pair plot using seaborn
# scatter plot heatmap
# contour plot
# line plot for performance

def visualizeData(data):
    sb.pairplot(data, hue='quality', diag_kind='hist')    
