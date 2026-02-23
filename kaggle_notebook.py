import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import warnings
import os

warnings.filterwarnings('ignore')

# ==================== LOAD DATA ====================
input_path = '/kaggle/input/cmi-detect-behavior-with-sensor-data'
train_df = pd.read_csv(f'{input_path}/train.csv')
test_df = pd.read_csv(f'{input_path}/test.csv')
print(f'Train: {train_df.shape} | Test: {test_df.shape}')

# ==================== UTILITY FUNCTIONS ====================
def remove_gravity_from_acc(acc_x, acc_y, acc_z, rot_x, rot_y, rot_z, rot_w):
    rotations = Rotation.from_quat(np.column_stack([rot_x, rot_y, rot_z, rot_w]))
    gravity = np.array([0, 0, -9.81])
    gravity_body = rotations.inv().apply(gravity)
    return acc_x - gravity_body[:, 0], acc_y - gravity_body[:, 1], acc_z - gravity_body[:, 2]

def fill_missing_values(df):
    return df.ffill().bfill().fillna(0)

def replace_tof_missing(df):
    tof_cols = [col for col in df.columns if 'tof' in col.lower()]
    df[tof_cols] = df[tof_cols].replace(-1, 500)
    return df

# ==================== PREPROCESSING ====================
class Preprocessor:
    def __init__(self):
        self.problematic_subjects = ['SUBJ_045235', 'SUBJ_019262']
        self.seq_without_gesture = 'SEQ_011975'
    
    def clean_tof(self, df):
        return replace_tof_missing(df)
    
    def fill_values(self, df):
        return fill_missing_values(df)
    
    def pad_sequence(self, df, max_length=120):
        if len(df) > max_length:
            return df.iloc[:max_length]
        elif len(df) < max_length:
            pad_length = max_length - len(df)
            padding = pd.concat([df.iloc[-1:]] * pad_length, ignore_index=True)
            return pd.concat([df, padding], ignore_index=True)
        return df

# ==================== FEATURE ENGINEERING ====================
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

# ==================== PIPELINE ====================
print("Filtrando dados...")
df_filtered = train_df[train_df.get('sequence_id', '') != 'SEQ_011975'].copy()

if 'gesture' in df_filtered.columns:
    seq_ids = []
    for seq_id in df_filtered['sequence_id'].unique():
        seq_data = df_filtered[df_filtered['sequence_id'] == seq_id]
        ratio = (seq_data['gesture'] != 0).sum() / len(seq_data)
        if ratio >= 0.2:
            seq_ids.append(seq_id)
    df_filtered = df_filtered[df_filtered['sequence_id'].isin(seq_ids)]

print("Pré-processando...")
preprocessor = Preprocessor()
sequences = []
for seq_id in df_filtered['sequence_id'].unique():
    seq = df_filtered[df_filtered['sequence_id'] == seq_id].reset_index(drop=True)
    seq = preprocessor.clean_tof(seq)
    seq = preprocessor.fill_values(seq)
    seq = preprocessor.pad_sequence(seq, 120)
    sequences.append(seq)

df_preprocessed = pd.concat(sequences, ignore_index=True)

print("Criando features...")
feature_engineer = FeatureEngineer()
sequences_features = []
for seq_id in df_preprocessed['sequence_id'].unique():
    seq = df_preprocessed[df_preprocessed['sequence_id'] == seq_id]
    try:
        features = feature_engineer.create_imu_features(seq)
        if 'gesture' in seq.columns:
            features['gesture'] = seq['gesture'].values
        sequences_features.append(features)
    except:
        pass

df_features = pd.concat(sequences_features, ignore_index=True)
print(f"Features: {df_features.shape}")

# ==================== TRAINING ====================
print("Treinando modelo com GridSearchCV e validação cruzada k-fold=5...")
feature_cols = [col for col in IMU_FEATURES if col in df_features.columns]
X = df_features[feature_cols].fillna(0)
y = df_features['gesture']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo o grid de hiperparâmetros para busca
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [8, 10, 12, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# GridSearchCV com k-fold = 5
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Exibindo os melhores hiperparâmetros encontrados
print("\n" + "="*60)
print("MELHORES HIPERPARÂMETROS ENCONTRADOS:")
print("="*60)
best_params = grid_search.best_params_
print(f"n_estimators: {best_params['n_estimators']}")
print(f"max_depth: {best_params['max_depth']}")
print(f"min_samples_split: {best_params['min_samples_split']}")
print(f"min_samples_leaf: {best_params['min_samples_leaf']}")
print(f"max_features: {best_params['max_features']}")
print(f"Melhor score (CV): {grid_search.best_score_:.4f}")
print("="*60 + "\n")

# Usando o melhor modelo
model = grid_search.best_estimator_

# Avaliando o modelo
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train Score: {train_score:.4f} | Test Score: {test_score:.4f}")
print(f"Diferença (overfitting check): {train_score - test_score:.4f}")

# ==================== PREDICTION ====================
print("Gerando submissão...")
test_sequences = []
for seq_id in test_df['sequence_id'].unique():
    seq = test_df[test_df['sequence_id'] == seq_id].reset_index(drop=True)
    seq = preprocessor.clean_tof(seq)
    seq = preprocessor.fill_values(seq)
    seq = preprocessor.pad_sequence(seq, 120)
    test_sequences.append(seq)

df_test_preprocessed = pd.concat(test_sequences, ignore_index=True)
test_features_list = []
for seq_id in df_test_preprocessed['sequence_id'].unique():
    seq = df_test_preprocessed[df_test_preprocessed['sequence_id'] == seq_id]
    try:
        features = feature_engineer.create_imu_features(seq)
        test_features_list.append(features)
    except:
        pass

df_test_features = pd.concat(test_features_list, ignore_index=True)
predictions = model.predict(df_test_features[feature_cols].fillna(0))

submission = pd.DataFrame({
    'index': range(len(predictions)),
    'gesture': predictions
})

submission.to_csv('submission.csv', index=False)
print(f"✅ Submissão: {submission.shape[0]} linhas")
print(submission.head(10))
