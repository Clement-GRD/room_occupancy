from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
class Window_Average (BaseEstimator,TransformerMixin):
    """
    Returns the averaged dataframe averaged using a rolling windows running along axis = 0 with the specified window size.
    """
    def __init__(self, window_size=1):
        self.window_size = window_size

    def fit(self, X, y=None):
        # Fitting does not change the state of the estimator
        return self

    def transform(self, X, y=None):
        # transform only transforms the data
        return X.rolling(self.window_size,min_periods=1).mean()
    
class Window_Average_w_Edges (BaseEstimator,TransformerMixin):
    """
    Returns the averaged dataframe averaged using a rolling windows running along axis = 0 with the specified window size.
    Takes into account the index to only apply the rolling average for instances having consecutive index.
    If an index gap is created during a train/test split, a different window average will be used for the different segments.
    """
    def __init__(self, window_size=1):
        self.window_size = window_size

    def fit(self, X, y=None):
        # Fitting does not change the state of the estimator
        return self

    def transform(self, X, y=None):
        # transform only transforms the data
        new_df = X.copy()
        index_group = pd.Series(np.cumsum(X.index.diff(1).fillna(1)!=1),index=X.index) #create a series containing the same number for groups of data with consecutive index    
        for i in index_group.unique():
            new_df[index_group==i] = (X[index_group==i]).rolling(self.window_size,min_periods=1).mean()
        return new_df
