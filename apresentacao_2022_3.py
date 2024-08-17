import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st

# Função para carregar o modelo e o escalador para 2022
def carregar_modelo_e_scaler():
    model_path = 'https://github.com/Henitz/fase5/raw/main/modelo.h5'
    scaler_path = 'https://github.com/Henitz/fase5/raw/main/scaler.pkl'

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
