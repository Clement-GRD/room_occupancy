import pandas as pd

file_path = 'https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip'

def main(file_path):
    df = read_file_df(file_path)
    print(df.head())

def read_file_df(file_path) :
    """ fetches zipped csv file from URL"""
    df  = pd.read_csv(file_path, compression='zip', header=0, sep=',', quotechar='"')
    return df

if __name__ == '__main__' :
    main(file_path)