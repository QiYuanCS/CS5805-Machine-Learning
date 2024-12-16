import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

class DataPreprocessing:
    def __init__(self, data):
        self.data = data
        self.normalized_data = None
        self.standardized_data = None
        self.iqr_scaled_data = None

    def normalize(self):
        scaler = MinMaxScaler().fit(self.data)
        self.normalized_data = pd.DataFrame(scaler.transform(self.data), columns = self.data.columns)

    def standardize(self):
        scaler = StandardScaler().fit(self.data)
        self.standardized_data = pd.DataFrame(scaler.transform(self.data), columns = self.data.columns)

    def iqr(self):
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        self.iqr_scaled_data = (self.data - Q1) / IQR

    def show_original(self, ax):
        self.data.plot(ax=ax)
        ax.set_title('Original APPL data set')
        ax.set_xlabel('Date')
        ax.legend(self.data.columns)

    def show_normalized(self, ax):
        if self.normalized_data is None:
            self.normalize()
        self.normalized_data.plot(ax=ax)
        ax.set_title('Normalized APPL data set')
        ax.set_xlabel('Date')
        ax.legend(self.normalized_data.columns)

    def show_standardized(self, ax):
        if self.standardized_data is None:
            self.standardize()
        self.standardized_data.plot(ax=ax)
        ax.set_title('Standardized APPL data set')
        ax.set_xlabel('Date')
        ax.legend(self.standardized_data.columns)

    def show_iqr(self, ax):
        if self.iqr_scaled_data is None:
            self.iqr_transform()
        self.iqr_scaled_data.plot(ax=ax)
        ax.set_title('IQR transformation APPL data set')
        ax.set_xlabel('Date')
        ax.legend(self.iqr_scaled_data.columns)