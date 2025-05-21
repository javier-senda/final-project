import json
import joblib
import streamlit as st
from transformers import pipeline
import pandas as pd

# Título de la app
st.title("Aplicación de predicción de salario en LinkedIn")

# Cargar modelos con manejo de errores
try:
    nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except Exception as e:
    st.error(f"❌ Error al cargar el modelo NLP: {e}")
    st.stop()

try:
    xgboost_model = joblib.load('mejor_modelo.pkl')
except Exception as e:
    st.error(f"❌ Error al cargar el modelo XGBoost: {e}")
    st.stop()

# Cargar etiquetas
try:
    with open('labels.json', 'r') as f:
        labels_dict = json.load(f)
except Exception as e:
    st.error(f"❌ Error al cargar el archivo labels.json: {e}")
    st.stop()

# Validar que labels_dict tenga contenido esperado
if not isinstance(labels_dict, dict) or not labels_dict:
    st.error("❌ El archivo labels.json no tiene el formato esperado.")
    st.stop()

# Función para procesar la descripción
def obtener_informacion_descripcion(descripcion, labels):
    result_dict = {}
    for category, phrases in labels.items():
        try:
            result = nlp_model(descripcion, candidate_labels=phrases)
            score = max(result["scores"])  # score más alto para esa categoría
        except Exception as e:
            st.warning(f"⚠️ Error procesando categoría '{category}': {e}")
            score = 0.0
        result_dict[category] = [score]
    return pd.DataFrame(result_dict)

# Función para predecir el salario
def predecir_salario(df_unificado, xgboost_model):
    try:
        return xgboost_model.predict(df_unificado)
    except Exception as e:
        st.error(f"❌ Error al predecir con el modelo XGBoost: {e}")
        return [0.0]

# Entrada del usuario
descripcion = st.text_area("✏️ Pega la descripción de la oferta de trabajo:")

# Validación de entrada
if descripcion:
    descripcion = descripcion.strip()
    if len(descripcion) < 30:
        st.warning("⚠️ La descripción es demasiado corta. Por favor proporciona más detalles.")
    else:
        # Procesar la descripción y predecir
        nlp_df = obtener_informacion_descripcion(descripcion, labels_dict)
        prediccion = predecir_salario(nlp_df, xgboost_model)
        st.success(f"💰 Salario predicho: {prediccion[0]:,.2f}")
else:
    st.info("📌 Introduce una descripción de oferta de trabajo para obtener la predicción.")
