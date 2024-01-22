import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score,accuracy_score,ConfusionMatrixDisplay

def read_file_df(file_path) :
    """ 
    Fetches zipped csv file from URL
    Parses date from 'Date' and 'Time' columns
    Returns dataframe containing data
    """
    df  = pd.read_csv(file_path, compression='zip', parse_dates={"date": ["Date", "Time"]}, header=0, sep=',', quotechar='"')
    return df

def plot_onecolumn_index_occupancy (df, column):
    """ 
    Plots the selected column vs index, with Room_Occupancy as the hue.
    """
    fig,ax = plt.subplots(1,1,figsize = (15,5))
    sns.scatterplot(data = df, x=df.index, y=column,hue = 'Room_Occupancy_Count', palette='coolwarm', edgecolor="none")
    ax.set_xlabel('index')

def plot_columns_index_occupancy (df, columns):
    """ 
    Plots the selected columns on separate graphs, sharing the same x (index), with Room_Occupancy as the hue.
    """
    fig,axs = plt.subplots(len(columns),1,figsize = (15,10),sharex=True)
    fig.subplots_adjust(hspace=0)
    for i,col in enumerate(columns) :
        sns.scatterplot(data = df, x=df.index, y=col, ax=axs[i],hue = 'Room_Occupancy_Count', palette='coolwarm', edgecolor="none",legend=i==0)
    axs[-1].set_xlabel('index')

def plot_one_columns_index_occupancy_compare_transformation (df, column,transformation):
    """ 
    Plots the selected column before and after transformation on separate graphs. The dataframe index is used as x, with Room_Occupancy as the hue.
    """
    _,axs = plt.subplots(1,2,figsize = (10,5))
    sns.scatterplot(ax = axs[0],x = df.index,y = df[column], hue =df['Room_Occupancy_Count'],palette='coolwarm',edgecolor='none', legend=False)
    sns.scatterplot(ax = axs[-1],x = df.index,y = transformation.fit_transform(df[column]), hue=df['Room_Occupancy_Count'],palette='coolwarm',edgecolor='none')

def print_metrics(estimator,cross_validate):
    """
    Prints the min,mean and max values of Accuracy (A) and F1-score macro (F1) estimated with the given cross_validate procedure and estimator.
    """
    print(estimator)
    print('metrics\t min\t mean\t max')
    print('A', '\t',np.round(np.min(cross_validate['test_accuracy']),3),'\t', np.round(np.mean(cross_validate['test_accuracy']),3),'\t',np.round(np.max(cross_validate['test_accuracy']),3))
    print('F1', '\t',np.round(np.min(cross_validate['test_f1_macro']),3),'\t', np.round(np.mean(cross_validate['test_f1_macro']),3),'\t',np.round(np.max(cross_validate['test_f1_macro']),3))

def plot_confusion_acc_f1 (y_true,y_test,title):
    """
    Plots the confusion matrix base on prediction with the given title.
    Also return accuracy and F1-score macro average.
    """
    ConfusionMatrixDisplay.from_predictions(y_true,y_test)
    plt.title(title)
    plt.xlabel('Predicted label\n Accuracy = {:0.3f}  F1-macro = {:0.3f}'.format(accuracy_score(y_true,y_test),f1_score(y_true,y_test,average='macro')))
    plt.plot();

if __name__ == '__main__' :
    main(file_path)