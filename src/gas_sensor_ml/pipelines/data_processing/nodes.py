"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.14
"""
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def _hi_lo_peak(x: pd.DataFrame) -> pd.DataFrame:
    peaks, properties = find_peaks(x['A1_Sensor'], width=50, height=1)
    peak_heights = properties['peak_heights']
# Determine smaller and larger peaks
    smaller_peaks, larger_peaks = [], []
    for i in range(len(peaks) - 1):
        if peak_heights[i] > peak_heights[i + 1]:
            larger_peaks.append(peaks[i])
            smaller_peaks.append(peaks[i + 1])
    # smaller_peaks_df = x.iloc[smaller_peaks]
    return smaller_peaks

def data_stack(sp: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    After finding the peaks, stack the data according to exp_no
    """
    df_stacked_list = []
    for i in range(len(sp) - 1):
        df_subset = df.iloc[sp[i]:sp[i + 1]].copy()
        df_subset['exp_no'] = i
        df_subset['timestamp'] -= df_subset['timestamp'].iloc[0]
        df_stacked_list.append(df_subset)
        df_stacked = pd.concat(df_stacked_list, ignore_index=True)
    return df_stacked


def _group_by_bin(df_stacked: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    """
    Use PD.CUT to group data into specified bins in parameters
    """
    df_list = []
    grouped = df_stacked.groupby('exp_no')
    for name, group in grouped:
        group['bin'] = pd.cut(group['timestamp'], bins=num_bins, labels=False)
        df_list.append(group)
    return pd.concat(df_list)

def _average_bin(bin_df: pd.DataFrame) -> pd.DataFrame:
    """
    average values within each bin to return only one data point
    """
    bin_df = bin_df.drop(columns=['timestamp'])
    grouped = bin_df.groupby(['exp_no', 'bin']).mean()
    return grouped.reset_index()

def preprocess_data_bin(mox: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    """
    Return data that is sorted by experiment number according to lo_peak interval
    data is stacked and labeled by exp_no
    data is grouped by bin and averaged
    """
    df_stacked = data_stack(_hi_lo_peak(mox), mox)
    bin_df = _group_by_bin(df_stacked, num_bins)
    mean_bin = _average_bin(bin_df)
    return mean_bin

def get_percentile_data(df, percentile):
    """
    Returns the data up to the specified percentile based on the 'bin' column.

    :param df: DataFrame containing the data
    :param percentile: A float value between 0 and 1 representing the percentile
    :return: DataFrame containing the data up to the specified percentile
    """
    # Calculate the bin index corresponding to the percentile
    max_bin = int(percentile * df['bin'].max())

    # Return data up to that bin
    return df[df['bin'] <= max_bin]

def _group_percentile (averaged: pd.DataFrame, percentile_bins: float) -> pd.DataFrame:
    """
    Returns the full specified percentile dataset
    """
    df_list = []
    grouped = averaged.groupby('exp_no')
    for name, group in grouped:
        percentile_data = get_percentile_data(group, percentile_bins)
        df_list.append(percentile_data)
    return pd.concat(df_list)

def _transpose_(df_set: pd.DataFrame) -> pd.DataFrame:
    transposed = df_set.pivot(index='exp_no', columns='bin', values='A1_Resistance')
    transposed.columns = ['bin_' + str(col) for col in transposed.columns]
    transposed.reset_index(inplace=True)
    return transposed


def _res_ratio(averaged: pd.DataFrame) -> pd.DataFrame:
    def calculate_res_ratio(group):
        return group['A1_Resistance'].max() / group['A1_Resistance'].min()

    res_ratio = averaged.groupby('exp_no').apply(calculate_res_ratio).reset_index()
    res_ratio.columns = ['exp_no', 'res_ratio']
    return res_ratio

def _combine_feature_matrix(res_ratio: pd.DataFrame, transposed: pd.DataFrame) -> pd.DataFrame:
    combined = pd.merge(res_ratio, transposed, on='exp_no')
    return combined

def create_model_input_table(mox_bin: pd.DataFrame, percentile_bins: float) -> pd.DataFrame:
    selected_range = _group_percentile(mox_bin, percentile_bins)
    # the ratio is from the entire dataset not filtered to be ground truth
    res_ratio = _res_ratio(mox_bin) 
    transpose_col = _transpose_(selected_range)
    # drop exp_no to avoid training on exp_no
    mox_table = _combine_feature_matrix(transpose_col, res_ratio).drop(columns=['exp_no'])
    return mox_table