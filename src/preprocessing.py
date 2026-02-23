import numpy as np
import pandas as pd
from .utils import remove_gravity_from_acc, mirror_left_handed_data, fill_missing_values, replace_tof_missing


class Preprocessor:
    
    def __init__(self):
        self.problematic_subjects = ['SUBJ_045235', 'SUBJ_019262']
        self.seq_without_gesture = 'SEQ_011975'
    
    def filter_gesture_ratio(self, df, min_ratio=0.2):
        if 'gesture' not in df.columns:
            return df
        ratio = (df['gesture'] != 0).sum() / len(df)
        return df if ratio >= min_ratio else None
    
    def remove_bad_sequences(self, df, remove_subjects=False):
        if 'sequence_id' in df.columns:
            if df['sequence_id'].iloc[0] == self.seq_without_gesture:
                return None
        
        if remove_subjects and 'subject_id' in df.columns:
            if df['subject_id'].iloc[0] in self.problematic_subjects:
                return None
        
        return df
    
    def clean_tof(self, df):
        return replace_tof_missing(df)
    
    def fill_values(self, df):
        return fill_missing_values(df)
    
    def mirror_left_handed(self, df, is_left=False):
        return mirror_left_handed_data(df) if is_left else df
    
    def pad_sequence(self, df, max_length=120):
        if len(df) > max_length:
            return df.iloc[:max_length]
        elif len(df) < max_length:
            pad_length = max_length - len(df)
            padding = pd.concat([df.iloc[-1:]] * pad_length, ignore_index=True)
            return pd.concat([df, padding], ignore_index=True)
        return df

