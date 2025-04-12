
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.colab import drive
import os
import requests

# Montar Google Drive (esto es solo para Colab, no es necesario en una instalación local)
# drive.mount('/content/drive')  # Descomenta si lo estás ejecutando en Google Colab

# Función para cargar el modelo desde Google Drive
def load_model_from_drive(file_id): 1dEqzErkWy7DH5ctXLjDyaEbpgREKKMOw
    model_url = f"https://drive.google.com/uc?id={file_id}"
    model_path = "random_survival_forest_model.pkl"
    
    # Descargar el modelo desde Google Drive
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)
    
    # Cargar el modelo
    model = joblib.load(model_path)
    return model

# Reemplaza "your-file-id" con el ID real del archivo de Google Drive
file_id = 'your-file-id'  # Pega aquí el ID de tu archivo en Google Drive
rsf = load_model_from_drive(file_id)

# Función de mapeo para las variables categóricas
def codificar_variables(paciente):
    paciente_codificado = paciente.copy()
    
    # Codificar variables categóricas (simular el mapeo que se hizo en el modelo)
    paciente_codificado['Sex_Male'] = 1 if paciente['Sex_Male'] == 'Male' else 0
    paciente_codificado['Transmission_mode_Homo/Bisexual'] = 1 if paciente['Transmission_mode_Homo/Bisexual'] == 'Homo/Bisexual' else 0
    paciente_codificado['Origin_Spain'] = 1 if paciente['Origin_Spain'] == 'Spain' else 0
    paciente_codificado['Education_Level_University'] = 1 if paciente['Education_Level_University'] == 'University' else 0
    paciente_codificado['AIDS_Yes'] = 1 if paciente['AIDS_Yes'] == 'Yes' else 0
    paciente_codificado['Viral_load_<100000'] = 1 if paciente['Viral_load_<100000'] == '<100000' else 0
    paciente_codificado['ART_2NRTI+1NNRTI'] = 1 if paciente['ART_2NRTI+1NNRTI'] == '2NRTI+1NNRTI' else 0
    paciente_codificado['HCV_Antibodies_Positive'] = 1 if paciente['HCV_Antibodies_Positive'] == 'Positive' else 0
    paciente_codificado['HBV_Anticore_Positive'] = 1 if paciente['HBV_Anticore_Positive'] == 'Positive' else 0
    paciente_codificado['HBP_Yes'] = 1 if paciente['HBP_Yes'] == 'Yes' else 0
    paciente_codificado['Smoking_Current'] = 1 if paciente['Smoking_Current'] == 'Current' else 0
    paciente_codificado['Diabetes_Yes'] = 1 if paciente['Diabetes_Yes'] == 'Yes' else 0
    
    return paciente_codificado

# Función para calcular la probabilidad de ECV
def calcular_probabilidad_ecv(paciente):
    paciente_codificado = codificar_variables(paciente)
    
    # Alinear el paciente con las columnas del modelo
    paciente_df = pd.DataFrame([paciente_codificado])
    paciente_df_alineado = paciente_df.reindex(columns=X.columns, fill_value=0)
    
    # Predecir la probabilidad de supervivencia
    predicciones = rsf.predict_survival_function(paciente_df_alineado)
    
    # Calcular el riesgo de ECV (probabilidad de supervivencia)
    riesgo = predicciones[0].y[abs(predicciones[0].x - 5).argmin()]  # A los 5 años
    riesgo_ecv = 1 - riesgo
    
    return riesgo_ecv * 100  # Convertir a porcentaje

# Crear la interfaz de usuario en Streamlit
st.title('Predicción de Riesgo de Evento Cardiovascular (ECV)')

# Ingresar las variables del paciente tipo
paciente = {
    'Age': st.slider('Edad', 18, 100, 45),
    'CD4_nadir': st.slider('CD4 nadir', 50, 1200, 350),
    'CD8_nadir': st.slider('CD8 nadir', 100, 1200, 800),
    'CD4_CD8_ratio': st.slider('Cociente CD4/CD8', 0.1, 2.0, 0.44),
    'Cholesterol_Total': st.slider('Colesterol total', 100, 300, 200),
    'HDL': st.slider('HDL', 30, 100, 60),
    'Triglycerides': st.slider('Triglicéridos', 50, 300, 150),
    'Non_HDL': st.slider('Colesterol no HDL', 50, 200, 140),
    'Trig_HDL_ratio': st.slider('Relación triglicéridos/HDL', 0.5, 5.0, 2.5),
    'Sex_Male': st.selectbox('Sexo', ['Male', 'Female']),
    'Transmission_mode_Homo/Bisexual': st.selectbox('Modo de transmisión', ['Homo/Bisexual', 'IDU', 'Heterosexual', 'Other/Unknown']),
    'Origin_Spain': st.selectbox('Origen', ['Spain', 'Not Spain']),
    'Education_Level_University': st.selectbox('Nivel educativo', ['University', 'Other']),
    'AIDS_Yes': st.selectbox('Diagnóstico de sida', ['Yes', 'No']),
    'Viral_load_<100000': st.selectbox('Carga viral', ['<100000', '>=100000']),
    'ART_2NRTI+1NNRTI': st.selectbox('Tratamiento ART', ['2NRTI+1NNRTI', 'Other']),
    'HCV_Antibodies_Positive': st.selectbox('Serología VHC', ['Positive', 'Negative']),
    'HBV_Anticore_Positive': st.selectbox('Serología anticore VHB', ['Positive', 'Negative']),
    'HBP_Yes': st.selectbox('Hipertensión', ['Yes', 'No']),
    'Smoking_Current': st.selectbox('Tabaquismo', ['Current', 'No']),
    'Diabetes_Yes': st.selectbox('Diabetes', ['Yes', 'No'])
}

# Calcular la probabilidad de ECV para el paciente tipo
probabilidad_ecv = calcular_probabilidad_ecv(paciente)

# Visualizar la probabilidad de ECV con una escala de colores
fig, ax = plt.subplots(figsize=(8, 1))
if probabilidad_ecv < 5:
    color = 'green'
elif 5 <= probabilidad_ecv <= 10:
    color = 'orange'
else:
    color = 'red'
    
ax.barh([0], [probabilidad_ecv], color=color)
ax.set_xlim(0, 15)
ax.set_yticks([])
ax.set_xlabel('Probabilidad de ECV (%)')
ax.set_title('Escala de riesgo de ECV a los 5 años')

for bar in ax.patches:
    width = bar.get_width()
    ax.text(width - 0.5, bar.get_y() + bar.get_height() / 2, f'{width:.2f}%', va='center', ha='right', color='black')

# Agregar una barra de color (escala) que va de 0 a 15%
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0, vmax=15))
sm.set_array([])  # Para que la barra de color funcione
fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05)

plt.tight_layout()
plt.show()


