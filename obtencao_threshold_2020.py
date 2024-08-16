import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import joblib
from collections import Counter

# Configurações iniciais
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Caminho do diretório da aplicação
root_dir = os.path.abspath(os.path.dirname(__file__))

# Caminhos para o scaler e o modelo
scaler_filename = os.path.join(root_dir, 'scaler_2022.pkl')
model_filename = os.path.join(root_dir, 'modelo_2022.h5')

# Depuração: imprimir os caminhos para verificar
print(f"Scaler path: {scaler_filename}")
print(f"Model path: {model_filename}")

# Verificar se os arquivos existem e são acessíveis
try:
    with open(scaler_filename, 'rb') as f:
        print("Scaler file opened successfully.")
        # Teste de carregamento
        loaded_scaler = joblib.load(f)
        print("Scaler carregado com sucesso.")
except FileNotFoundError:
    print("Scaler file not found.")
except Exception as e:
    print(f"Erro ao carregar scaler: {e}")

try:
    # Carregando o modelo TensorFlow
    loaded_model = tf.keras.models.load_model(model_filename)
    print("Modelo carregado com sucesso.")
except FileNotFoundError:
    print("Model file not found.")
except Exception as e:
    print(f"Erro ao carregar modelo: {e}")

# Se ambos os arquivos foram carregados com sucesso, continuar o processamento
if 'loaded_scaler' in locals() and 'loaded_model' in locals():
    # Carregar o dataset
    file_path = os.path.join(root_dir, 'PEDE_PASSOS_DATASET_FIAP.csv')
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip')

    # Remover todas as linhas com qualquer valor NaN em qualquer coluna
    df = df.dropna()

    # Selecionar colunas numéricas para 2020
    caracteristicas_numericas = [
        'IAA_2022', 'IEG_2022', 'IPS_2022',
        'IDA_2022', 'IPP_2022', 'IPV_2022', 'IAN_2022',
        'NOTA_MAT_2022', 'NOTA_PORT_2022', 'NOTA_ING_2022'
    ]

    # Filtrar o DataFrame para as colunas de 2022
    colunas_numericas_existentes = [col for col in caracteristicas_numericas if col in df.columns]
    df_2022 = df[colunas_numericas_existentes + ['PONTO_VIRADA_2022']]

    # Verificar o desequilíbrio de classes
    print(Counter(df_2022['PONTO_VIRADA_2022']))

    # Definir características e alvo
    X = df_2022[colunas_numericas_existentes]
    y = df_2022['PONTO_VIRADA_2022']

    # Converter y para valores inteiros usando LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Normalizar os dados
    X_normalized = loaded_scaler.transform(X)

    # Fazer previsões
    probas = loaded_model.predict(X_normalized).flatten()

    # Encontrar o melhor threshold
    best_threshold = 0.0
    best_f1 = 0.0

    for threshold in np.arange(0.0, 1.0, 0.01):
        ponto_virada = (probas > threshold).astype(int)
        f1 = f1_score(y, ponto_virada, zero_division=1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Melhor threshold: {best_threshold}, Melhor F1 Score: {best_f1}")

    # Salvar os resultados
    resultados_df = pd.DataFrame({
        'Probabilidades': probas,
        'Ponto_Virada_Previsto': (probas > best_threshold).astype(int),
        'Ponto_Virada_Real': y
    })
    resultados_df['Acertos'] = resultados_df['Ponto_Virada_Previsto'] == resultados_df['Ponto_Virada_Real']
    output_path = os.path.join(root_dir, 'resultados_2020.csv')
    resultados_df.to_csv(output_path, index=False)
    print(f"Resultados salvos em {output_path}")
else:
    raise FileNotFoundError("Modelo ou scaler não encontrados.")
