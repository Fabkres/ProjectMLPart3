# CMI Detect Behavior with Sensor Data

Machine Learning para Kaggle: CMI Detect Behavior with Sensor Data

## Instalacao

```bash
pip install -r requirements.txt
kaggle auth login
mkdir -p data/raw
kaggle competitions download -c cmi-detect-behavior-with-sensor-data -p data/raw
```

## Uso

```bash
python -m src.pipeline
```

## Estrutura

- src/utils.py - Funcoes auxiliares
- src/preprocessing.py - Pre-processamento
- src/feature_engineering.py - 35 features de IMU
- src/pipeline.py - Pipeline principal
- config.py - Configuracoes

## Processamento

1. Download de dados Kaggle
2. Filtragem de sequencias ruins
3. Pre-processamento e normalizacao
4. Engenharia de 35 features
5. Treinamento XGBoost
6. Geracao de submissao

## Features (35 no total)

- Aceleracao: acc_x, acc_y, acc_z
- Magnitude: acc_mag, acc_mag_jerk
- Jerk temporal: jerk_x, jerk_y, jerk_z, jerk_magnitude
- Correlacoes: acc_xy_corr, acc_xz_corr, acc_yz_corr
- Rotacao: rot_w, rot_x, rot_y, rot_z
- Angulos: rot_angle, rot_angle_vel
- Velocidade angular: angular_vel_x/y/z, magnitude, distance
- Sem gravidade: versoes 2x de acceleration, jerk e correlacoes
