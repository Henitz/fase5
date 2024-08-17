import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st
import requests


# Função para baixar um arquivo a partir de uma URL
def baixar_arquivo(url, nome_arquivo):
    response = requests.get(url)
    with open(nome_arquivo, 'wb') as f:
        f.write(response.content)


# Função para carregar o modelo e o escalador para 2022
def carregar_modelo_e_scaler():
    model_path = 'modelo.h5'
    scaler_path = 'scaler.pkl'

    # URLs dos arquivos no GitHub
    model_url = 'https://github.com/Henitz/fase5/raw/main/modelo_2022.h5'
    scaler_url = 'https://github.com/Henitz/fase5/raw/main/scaler_2022.pkl'

    # Baixar os arquivos do GitHub
    baixar_arquivo(model_url, model_path)
    baixar_arquivo(scaler_url, scaler_path)

    # Carregar o modelo
    try:
        model = tf.keras.models.load_model(model_path)
        print('Modelo TensorFlow para 2022 carregado com sucesso.')
    except Exception as e:
        print(f'Erro ao carregar o modelo para 2022: {e}')
        st.error(f'Erro ao carregar o modelo para 2022: {e}')
        return None, None

    # Carregar o escalador
    try:
        scaler = joblib.load(scaler_path)
        print('Scaler para 2022 carregado com sucesso.')
    except Exception as e:
        print(f'Erro ao carregar o escalador para 2022: {e}')
        st.error(f'Erro ao carregar o escalador para 2022: {e}')
        return None, None

    return model, scaler

# O restante do seu código continua igual...
