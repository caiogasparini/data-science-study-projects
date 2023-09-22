import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time # teste

class Preprocessor:
    def __init__(self) -> None:
        pass

    def heatpmap_correlation(self, df: pd.DataFrame) -> None:
        correlation = df.corr()
        corr_down = correlation.where(np.tril(np.ones(correlation.shape)).astype(bool))
        fig, ax = plt.subplots(figsize=[15,10])
        ax = sns.heatmap(corr_down, annot=True, fmt='.1f')
        plt.show()

    def feature_engeneering(self, series_1: pd.Series, series_2: pd.Series) -> pd.Series:
        s_normalized_1: pd.Series = self.__data_normalize(series_1, True)
        s_normalized_2: pd.Series = self.__data_normalize(series_2, True)
        s_ols_removed_1: pd.Series = self.__data_normalize(s_normalized_1, False)
        s_ols_removed_2: pd.Series = self.__data_normalize(s_normalized_2, False)
        df_dif: pd.Series = self.__get_df(s_ols_removed_1, s_ols_removed_2)
        df_dif.reset_index(drop=True, inplace=True)
        return df_dif

    def __data_normalize(self, series: pd.Series, remove_ols: bool) -> pd.Series:
        max: (float | int) = series.max()
        min: (float | int) = series.min()
        difference: (float | int) = max - min
        normalize: pd.Series = (series - min) / difference
        if remove_ols:
            s_ols_removed: pd.Series = self.__remove_outliers(normalize)
            return s_ols_removed
        else:
            return normalize

    def __remove_outliers(self, series: pd.Series) -> pd.Series:
        lower_std, upper_std = self.__get_upper_lower_std(series)
        s_median: float = series.median()
        start = time.time()
        series_no_outliers = [i if (i > lower_std) and (i < upper_std) else s_median for i in series]
        end = time.time()
        passed_time = end - start
        print(passed_time)
        return pd.Series(series_no_outliers, name=series.name)

    def __get_upper_lower_std(self, series: pd.Series) -> float:
        central_position: float = self.__get_central_position(series)
        std_2: float = series.std() * 2
        lower_std: float = central_position - std_2
        upper_std: float = central_position + std_2
        return lower_std, upper_std

    def __get_central_position(self, series: pd.Series) -> float:
        hist, bins = np.histogram(series, bins=40)
        max_bin_index = np.argmax(hist)
        max_bin: int = bins[max_bin_index]
        bin_len: float = bins[1] - bins[0]
        central_position: float = max_bin + (bin_len / 2)
        return central_position

    def __get_df(self, series_1: pd.Series, series_2: pd.Series) -> pd.Series:
        series_1.reset_index(drop=True, inplace=True)
        series_2.reset_index(drop=True, inplace=True)
        df_dif: pd.Series = series_1 - series_2
        return df_dif