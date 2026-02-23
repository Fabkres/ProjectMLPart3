import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import Preprocessor
from src.feature_engineering import FeatureEngineer
from config import (
    DATA_RAW, DATA_PROCESSED, FILTERS_CONFIG, PREPROCESSING_CONFIG,
    IMU_FEATURES, KAGGLE_CONFIG
)


class Pipeline:
    def __init__(self):
        self.data_path = DATA_RAW
        self.processed_path = DATA_PROCESSED
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        self.train_df = None
        self.preprocessor = Preprocessor()
    
    def download(self):
        print("Baixando dados Kaggle...")
        result = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', KAGGLE_CONFIG['competition_id'],
             '-p', str(self.data_path)],
            capture_output=True, text=True, timeout=300
        )
        return result.returncode == 0
    
    def load(self):
        print("Carregando dados...")
        files = list(self.data_path.glob('*.csv'))
        if not files:
            print("Nenhum arquivo CSV encontrado")
            return False
        
        train_file = self.data_path / 'train.csv'
        if train_file.exists():
            self.train_df = pd.read_csv(train_file)
            print(f"Train: {self.train_df.shape}")
        return self.train_df is not None
    
    def filter_data(self):
        print("Filtrando dados...")
        df = self.train_df
        
        # Remove sequencias ruins
        df = df[df.get('sequence_id', '') != FILTERS_CONFIG['remove_seq_without_gesture']]
        
        if 'gesture' in df.columns:
            seq_ids = []
            for seq_id in df['sequence_id'].unique():
                seq_data = df[df['sequence_id'] == seq_id]
                ratio = (seq_data['gesture'] != 0).sum() / len(seq_data)
                if ratio >= FILTERS_CONFIG['min_gesture_ratio']:
                    seq_ids.append(seq_id)
            df = df[df['sequence_id'].isin(seq_ids)]
        
        self.train_df = df
        print(f"Dados filtrados: {df.shape}")
        return True
    
    def preprocess(self):
        print("Pre-processando...")
        df = self.train_df.copy()
        
        # TOF -1 para 500
        df = self.preprocessor.clean_tof(df)
        
        # Preencher NaNs
        df = self.preprocessor.fill_values(df)
        
        # Padronizar comprimento
        sequences = []
        for seq_id in df['sequence_id'].unique():
            seq = df[df['sequence_id'] == seq_id].reset_index(drop=True)
            seq = self.preprocessor.pad_sequence(seq, PREPROCESSING_CONFIG['max_sequence_length'])
            sequences.append(seq)
        
        self.train_df = pd.concat(sequences, ignore_index=True)
        print(f"Pre-processado: {self.train_df.shape}")
        return True
    
    def create_features(self):
        print("Criando features...")
        
        sequences = []
        for seq_id in self.train_df['sequence_id'].unique():
            seq = self.train_df[self.train_df['sequence_id'] == seq_id]
            try:
                features = FeatureEngineer.create_imu_features(seq)
                if 'subject_id' in seq.columns:
                    features['subject_id'] = seq['subject_id'].iloc[0]
                if 'gesture' in seq.columns:
                    features['gesture'] = seq['gesture'].values
                sequences.append(features)
            except:
                pass
        
        if sequences:
            self.features_df = pd.concat(sequences, ignore_index=True)
            print(f"Features criadas: {self.features_df.shape}")
            return True
        return False
    
    def train(self):
        print("Treinando modelo...")
        
        df = self.features_df
        if 'gesture' not in df.columns:
            print("Coluna gesture nao encontrada")
            return False
        
        feature_cols = [col for col in IMU_FEATURES if col in df.columns]
        X = df[feature_cols].fillna(0)
        y = df['gesture']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Train accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
        self.model = model
        self.feature_cols = feature_cols
        return True
    
    def predict(self):
        print("Gerando submissao...")
        test_file = self.data_path / 'test.csv'
        if not test_file.exists():
            print("Arquivo test.csv nao encontrado")
            return False
        
        test_df = pd.read_csv(test_file)
        X_test = test_df[self.feature_cols].fillna(0)
        predictions = self.model.predict(X_test)
        
        submission = pd.DataFrame({
            'index': range(len(predictions)),
            'gesture': predictions
        })
        
        output = self.processed_path / 'submission.csv'
        submission.to_csv(output, index=False)
        print(f"Submissao salva: {output}")
        return True
    
    def run(self, download=False):
        print("\nCMI Pipeline")
        print("=" * 50)
        
        if download and not self.download():
            return False
        
        if not self.load():
            return False
        
        if not self.filter_data():
            return False
        
        if not self.preprocess():
            return False
        
        if not self.create_features():
            return False
        
        if not self.train():
            return False
        
        if not self.predict():
            return False
        
        print("=" * 50)
        print("Pipeline concluido com sucesso!")
        return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Baixar dados')
    args = parser.parse_args()
    
    pipeline = Pipeline()
    pipeline.run(download=args.download)
