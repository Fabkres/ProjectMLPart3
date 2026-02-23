import numpy as np
import pandas as pd
from .utils import remove_gravity_from_acc
from config import IMU_FEATURES


class FeatureEngineer:
    
    @staticmethod
    def create_imu_features(df):
        features = pd.DataFrame(index=df.index)
        
        features['acc_x'] = df.get('acc_x', 0)
        features['acc_y'] = df.get('acc_y', 0)
        features['acc_z'] = df.get('acc_z', 0)
        
        if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']):
            acc_x2, acc_y2, acc_z2 = remove_gravity_from_acc(
                df['acc_x'].values, df['acc_y'].values, df['acc_z'].values,
                df['rot_x'].values, df['rot_y'].values, df['rot_z'].values, df['rot_w'].values
            )
            features['acc_x2'] = acc_x2
            features['acc_y2'] = acc_y2
            features['acc_z2'] = acc_z2
        
        features['acc_mag'] = np.sqrt(features['acc_x']**2 + features['acc_y']**2 + features['acc_z']**2)
        
        features['jerk_x'] = features['acc_x'].diff().fillna(0)
        features['jerk_y'] = features['acc_y'].diff().fillna(0)
        features['jerk_z'] = features['acc_z'].diff().fillna(0)
        features['jerk_magnitude'] = np.sqrt(features['jerk_x']**2 + features['jerk_y']**2 + features['jerk_z']**2)
        features['acc_mag_jerk'] = features['acc_mag'].diff().fillna(0)
        
        window = 5
        features['acc_xy_corr'] = features['acc_x'].rolling(window).corr(features['acc_y']).fillna(0)
        features['acc_xz_corr'] = features['acc_x'].rolling(window).corr(features['acc_z']).fillna(0)
        features['acc_yz_corr'] = features['acc_y'].rolling(window).corr(features['acc_z']).fillna(0)
        
        features['rot_w'] = df.get('rot_w', 0)
        features['rot_x'] = df.get('rot_x', 0)
        features['rot_y'] = df.get('rot_y', 0)
        features['rot_z'] = df.get('rot_z', 0)
        
        features['rot_angle'] = 2 * np.arccos(np.clip(features['rot_w'], -1, 1))
        features['rot_angle_vel'] = features['rot_angle'].diff().fillna(0)
        
        features['angular_vel_x'] = df.get('angular_vel_x', features['rot_x'].diff().fillna(0))
        features['angular_vel_y'] = df.get('angular_vel_y', features['rot_y'].diff().fillna(0))
        features['angular_vel_z'] = df.get('angular_vel_z', features['rot_z'].diff().fillna(0))
        
        features['angular_vel_magnitude'] = np.sqrt(
            features['angular_vel_x']**2 + features['angular_vel_y']**2 + features['angular_vel_z']**2
        )
        features['angular_distance'] = features['rot_angle'].diff().fillna(0)
        
        features['acc_mag2'] = np.sqrt(features['acc_x2']**2 + features['acc_y2']**2 + features['acc_z2']**2)
        features['jerk_x2'] = features['acc_x2'].diff().fillna(0)
        features['jerk_y2'] = features['acc_y2'].diff().fillna(0)
        features['jerk_z2'] = features['acc_z2'].diff().fillna(0)
        features['jerk_magnitude2'] = np.sqrt(features['jerk_x2']**2 + features['jerk_y2']**2 + features['jerk_z2']**2)
        features['acc_mag_jerk2'] = features['acc_mag2'].diff().fillna(0)
        
        features['acc_xy_corr2'] = features['acc_x2'].rolling(window).corr(features['acc_y2']).fillna(0)
        features['acc_xz_corr2'] = features['acc_x2'].rolling(window).corr(features['acc_z2']).fillna(0)
        features['acc_yz_corr2'] = features['acc_y2'].rolling(window).corr(features['acc_z2']).fillna(0)
        
        return features[[col for col in IMU_FEATURES if col in features.columns]]

