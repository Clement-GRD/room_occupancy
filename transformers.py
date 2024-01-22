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