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
    df_stacked_list = []
    for i in range(len(sp) - 1):
        df_subset = df.iloc[sp[i]:sp[i + 1]].copy()
        df_subset['exp_no'] = i
        df_subset['timestamp'] -= df_subset['timestamp'].iloc[0]
        df_stacked_list.append(df_subset)
        df_stacked = pd.concat(df_stacked_list, ignore_index=True)
    return df_stacked

# def _group_by_bucket(df_stacked: pd.DataFrame, bucket_size_ms: int) -> pd.DataFrame:
#     df_list = []
#     grouped = df_stacked.groupby('exp_no')
#     for name, group in grouped:
#         group['timestamp_bucket'] = group['timestamp'].floordiv(bucket_size_ms)
#         df_list.append(group)
#     return pd.concat(df_list)

def _group_by_bin(df_stacked: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    df_list = []
    grouped = df_stacked.groupby('exp_no')
    for name, group in grouped:
        group['bin'] = pd.cut(group['timestamp'], bins=num_bins, labels=False)
        df_list.append(group)
    return pd.concat(df_list)

def _average_bin(bin_df: pd.DataFrame) -> pd.DataFrame:
    bin_df = bin_df.drop(columns=['timestamp'])
    grouped = bin_df.groupby(['exp_no', 'bin']).mean()
    return grouped.reset_index()

def preprocess_data_bin(mox: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    df_stacked = data_stack(_hi_lo_peak(mox), mox)
    bin_df = _group_by_bin(df_stacked, num_bins)
    mean_bin = _average_bin(bin_df)
    return mean_bin

def _transpose_(df_set: pd.DataFrame) -> pd.DataFrame:
    transposed = df_set.pivot(index='exp_no', columns='bin', values='A1_Resistance')
    transposed.columns = ['bin_' + str(col) for col in transposed.columns]
    transposed.reset_index(inplace=True)
    return transposed

def _remove_exp_no(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['exp_no'])
    return df

def create_model_input_table(mox_bin: pd.DataFrame, exposed_bins: int) -> pd.DataFrame:
    mox_table = _remove_exp_no(_transpose_(mox_bin)).iloc[:exposed_bins]
    return mox_table