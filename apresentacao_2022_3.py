import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st
import requests
import tempfile

# Forçar TensorFlow a usar apenas a CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
st.write("CPU mode ativado.")

# Função para baixar um arquivo a partir de uma URL e salvar em um arquivo temporário
def baixar_arquivo_temporario(url):
    st.write(f"Baixando arquivo de {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")  # Adicionando sufixo .h5
        with open(temp_file.name, 'wb') as f:
            f.write(response.content)
        st.write(f"Arquivo baixado e salvo em {temp_file.name} (Tamanho: {os.path.getsize(temp_file.name)} bytes)")
        return temp_file.name
    else:
        st.error(f"Erro ao baixar o arquivo: {response.status_code}")
        return None

# Função para carregar o modelo e o escalador para 2022
def carregar_modelo_e_scaler():
    st.write("Carregando modelo e scaler...")

    # URLs dos arquivos no GitHub
    model_url = 'https://github.com/Henitz/fase5/raw/master/modelo_2022.h5'  # Certifique-se que o URL tem .h5
    scaler_url = 'https://github.com/Henitz/fase5/raw/master/scaler_2022.pkl'

    # Baixar os arquivos do GitHub
    model_path = baixar_arquivo_temporario(model_url)
    scaler_path = baixar_arquivo_temporario(scaler_url)

    if model_path is None or scaler_path is None:
        st.error("Erro ao baixar os arquivos. Verifique as URLs.")
        return None, None

    # Carregar o modelo
    try:
        model = tf.keras.models.load_model(model_path)
        st.write('Modelo TensorFlow para 2022 carregado com sucesso.')
    except Exception as e:
        st.error(f'Erro ao carregar o modelo para 2022: {e}')
        return None, None

    # Carregar o escalador
    try:
        scaler = joblib.load(scaler_path)
        st.write('Scaler para 2022 carregado com sucesso.')
    except Exception as e:
        st.error(f'Erro ao carregar o escalador para 2022: {e}')
        return None, None

    return model, scaler

# Carregar modelo e scaler
model, scaler = carregar_modelo_e_scaler()

# Verificar se modelo e scaler foram carregados com sucesso
if model is None or scaler is None:
    st.write("Modelo ou scaler não foram carregados. Verifique os logs acima.")
else:
    st.write("Modelo e scaler carregados, pronto para fazer previsões.")

# Continuação das funcionalidades específicas da sua aplicação
# Adicione aqui as funções e o processamento adicional necessários
