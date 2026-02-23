import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def remove_gravity_from_acc(acc_x, acc_y, acc_z, rot_x, rot_y, rot_z, rot_w):
    rotations = Rotation.from_quat(np.column_stack([rot_x, rot_y, rot_z, rot_w]))
    gravity = np.array([0, 0, -9.81])
    gravity_body = rotations.inv().apply(gravity)
    
    acc_x_no_gravity = acc_x - gravity_body[:, 0]
    acc_y_no_gravity = acc_y - gravity_body[:, 1]
    acc_z_no_gravity = acc_z - gravity_body[:, 2]
    
    return acc_x_no_gravity, acc_y_no_gravity, acc_z_no_gravity


def mirror_left_handed_data(imu_data):
    data = imu_data.copy()
    
    if 'acc_x' in data.columns:
        data['acc_x'] = -data['acc_x']
    
    if all(col in data.columns for col in ['rot_w', 'rot_x', 'rot_y', 'rot_z']):
        rotations = Rotation.from_quat(
            data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
        )
        euler = rotations.as_euler('xyz', degrees=False)
        euler[:, 1] = -euler[:, 1]
        euler[:, 2] = -euler[:, 2]
        
        rotations_mirrored = Rotation.from_euler('xyz', euler, degrees=False)
        quat = rotations_mirrored.as_quat()
        
        data['rot_x'] = quat[:, 0]
        data['rot_y'] = quat[:, 1]
        data['rot_z'] = quat[:, 2]
        data['rot_w'] = quat[:, 3]
    
    return data


def fill_missing_values(df):
    return df.ffill().bfill().fillna(0)


def replace_tof_missing(df):
    tof_cols = [col for col in df.columns if 'tof' in col.lower()]
    if tof_cols:
        df[tof_cols] = df[tof_cols].replace(-1, 500)
    return df

