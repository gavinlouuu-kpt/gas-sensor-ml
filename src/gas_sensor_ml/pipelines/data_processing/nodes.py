"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.14
"""
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# def _hi_lo_peak(x: pd.DataFrame) -> pd.DataFrame:
#     peaks, properties = find_peaks(x['A1_Sensor'], width=50, height=1)
#     peak_heights = properties['peak_heights']
# # Determine smaller and larger peaks
#     smaller_peaks, larger_peaks = [], []
#     for i in range(len(peaks) - 1):
#         if peak_heights[i] > peak_heights[i + 1]:
#             larger_peaks.append(peaks[i])
#             smaller_peaks.append(peaks[i + 1])
#     return smaller_peaks

# def preprocess_data_stack(mox: pd.DataFrame) -> pd.DataFrame:
#     df_stacked_list = []
#     for i in range(len(_hi_lo_peak(mox)) - 1):
#         df_subset = mox.iloc[_hi_lo_peak(mox)[i]:_hi_lo_peak(mox)[i + 1]].copy()
#         df_subset['exp_no'] = i
#         df_subset['timestamp'] -= df_subset['timestamp'].iloc[0]
#         df_stacked_list.append(df_subset)
#         df_stacked = pd.concat(df_stacked_list, ignore_index=True)
#     return df_stacked

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

def _group_by_bucket(df_stacked: pd.DataFrame, bucket_size_ms: int) -> pd.DataFrame:
    df_list = []
    grouped = df_stacked.groupby('exp_no')
    for name, group in grouped:
        group['timestamp_bucket'] = group['timestamp'].floordiv(bucket_size_ms)
        df_list.append(group)
    return pd.concat(df_list)

def preprocess_data_bucket(mox: pd.DataFrame, bucket_size_ms: int) -> pd.DataFrame:
    df_stacked = data_stack(_hi_lo_peak(mox), mox)
    df_bucket = _group_by_bucket(df_stacked, bucket_size_ms)
    return df_bucket