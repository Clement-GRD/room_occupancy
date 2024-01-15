import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip'

def main(file_path):
    df = read_file_df(file_path)
    print(df.head())

def read_file_df(file_path) :
    """ 
    Fetches zipped csv file from URL
    Parses date from 'Date' and 'Time' columns
    Returns dataframe containing data
    """
    df  = pd.read_csv(file_path, compression='zip', parse_dates={"date": ["Date", "Time"]}, header=0, sep=',', quotechar='"')
    return df

def plot_columns_time_occupancy_2017 (df, columns):
    """ 
    Plots the selects columns as a function of time (for years before 2018), with Room_Occupancy as the hue.
    """
    fig,axs = plt.subplots(len(columns),1,figsize = (15,10),sharex=True)
    fig.subplots_adjust(hspace=0)
    for i,col in enumerate(columns) :
        sns.scatterplot(data = df.query('date < 2018'), x='date', y=col, alpha = 1, ax=axs[i],hue = 'Room_Occupancy_Count', palette='coolwarm', edgecolor="none",legend=i==0)

def print_metrics(estimator,cross_validate):
    """
    Prints the min,mean and max values of Accuracy (A) and F1-score macro (F1) estimated with the given cross_validate procedure and estimator.
    """
    print(estimator)
    print('metrics\t min\t mean\t max')
    print('A', '\t',np.round(np.min(cross_validate['test_accuracy']),3),'\t', np.round(np.mean(cross_validate['test_accuracy']),3),'\t',np.round(np.max(cross_validate['test_accuracy']),3))
    print('F1', '\t',np.round(np.min(cross_validate['test_f1_macro']),3),'\t', np.round(np.mean(cross_validate['test_f1_macro']),3),'\t',np.round(np.max(cross_validate['test_f1_macro']),3))

if __name__ == '__main__' :
    main(file_path)