import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
import joblib
from collections import Counter

# Configurações iniciais para minimizar logs desnecessários
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Caminho do diretório da aplicação
root_dir = os.path.abspath(os.path.dirname(__file__))

# Caminhos para o scaler e o modelo
scaler_filename = os.path.join(root_dir, 'scaler_2020.pkl')
model_filename = os.path.join(root_dir, 'modelo_2020.h5')

# Depuração: imprimir os caminhos absolutos para verificar
print(f"Scaler path: {scaler_filename}")
print(f"Model path: {model_filename}")

# Verificar se os arquivos existem
if os.path.exists(scaler_filename):
    print("Scaler encontrado.")
else:
    print("Scaler não encontrado. Verifique o caminho e a existência do arquivo.")

if os.path.exists(model_filename):
    print("Modelo encontrado.")
else:
    print("Modelo não encontrado. Verifique o caminho e a existência do arquivo.")

# Carregar o scaler e o modelo
try:
    loaded_scaler = joblib.load(scaler_filename)
    loaded_model = tf.keras.models.load_model(model_filename)
    print("Scaler e modelo carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar scaler ou modelo: {e}")
    raise FileNotFoundError("Modelo ou scaler não encontrados.")

# Carregar o dataset
file_path = './PEDE_PASSOS_DATASET_FIAP_sentimento2.csv'
pd.set_option('display.max_columns', None)
df = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip')

# Remover todas as linhas com qualquer valor NaN em qualquer coluna
df = df.dropna()

# Selecionar colunas numéricas para 2020
colunas_numericas_2020 = [
    'IDADE_ALUNO_2020', 'ANOS_PM_2020', 'INDE_2020',
    'IDA_2020',
    'IPP_2020', 'IPV_2020', 'IAN_2020',
    'DESTAQUE_IEG_2020_sentimento', 'DESTAQUE_IDA_2020_sentimento', 'DESTAQUE_IPV_2020_sentimento'
]

# Filtrar o DataFrame para as colunas de 2020
colunas_numericas_existentes = [col for col in colunas_numericas_2020 if col in df.columns]
df_2020 = df[colunas_numericas_existentes + ['PONTO_VIRADA_2020']]

# Verificar o desequilíbrio de classes
print(Counter(df_2020['PONTO_VIRADA_2020']))

# Converter y para valores inteiros usando LabelEncoder
le = LabelEncoder()
df_2020['PONTO_VIRADA_2020'] = le.fit_transform(df_2020['PONTO_VIRADA_2020'])
y_true = df_2020['PONTO_VIRADA_2020']

# Definir características e alvo
X = df_2020[colunas_numericas_existentes]

# Normalizar os dados
X_normalized = loaded_scaler.transform(X)

# Fazer previsões
probas = loaded_model.predict(X_normalized).flatten()

# Definir o threshold
threshold = 0.87
ponto_virada = (probas > threshold).astype(int)

# Comparar as previsões com os valores reais
acertos = (ponto_virada == y_true).sum()
erros = (ponto_virada != y_true).sum()

print(f"Número de acertos: {acertos}")
print(f"Número de erros: {erros}")

# Calcular métricas adicionais
f1 = f1_score(y_true, ponto_virada, zero_division=1)
accuracy = accuracy_score(y_true, ponto_virada)

print(f"F1 Score: {f1}")
print(f"Accuracy: {accuracy}")

# Salvar os resultados
resultados_df = pd.DataFrame({
    'Probabilidades': probas,
    'Ponto_Virada_Previsto': ponto_virada,
    'Ponto_Virada_Real': y_true
})
resultados_df['Acertos'] = resultados_df['Ponto_Virada_Previsto'] == resultados_df['Ponto_Virada_Real']

output_path = os.path.join(root_dir, 'resultados_threshold_2020.csv')
try:
    resultados_df.to_csv(output_path, index=False)
    print(f"Resultados salvos em {output_path}")
except PermissionError:
    print(f"Erro de permissão ao tentar salvar o arquivo em {output_path}. Verifique se o arquivo está aberto ou se você tem permissões suficientes.")
