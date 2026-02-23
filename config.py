from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

FILTERS_CONFIG = {
    'remove_seq_without_gesture': 'SEQ_011975',
    'min_gesture_ratio': 0.2,
    'tof_missing_threshold': 0.2,
    'problematic_subjects': ['SUBJ_045235', 'SUBJ_019262'],
}

PREPROCESSING_CONFIG = {
    'tof_missing_value': -1,
    'tof_replacement_value': 500,
    'sequence_percentile': 94,
    'max_sequence_length': 120,
}

IMU_FEATURES = [
    'acc_x', 'acc_y', 'acc_z', 'acc_mag', 'acc_mag_jerk',
    'jerk_x', 'jerk_y', 'jerk_z', 'jerk_magnitude',
    'acc_xy_corr', 'acc_xz_corr', 'acc_yz_corr',
    'rot_w', 'rot_x', 'rot_y', 'rot_z', 'rot_angle', 'rot_angle_vel',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_vel_magnitude', 'angular_distance',
    'acc_x2', 'acc_y2', 'acc_z2', 'acc_mag2', 'acc_mag_jerk2',
    'jerk_x2', 'jerk_y2', 'jerk_z2', 'jerk_magnitude2',
    'acc_xy_corr2', 'acc_xz_corr2', 'acc_yz_corr2',
]

KAGGLE_CONFIG = {
    'competition_id': 'cmi-detect-behavior-with-sensor-data',
    'test_filename': 'test.csv',
    'train_filename': 'train.csv',
}

