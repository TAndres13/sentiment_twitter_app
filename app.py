import streamlit as st
import joblib
import re
import nltk # Asegúrate de importar NLTK primero

# --- 🎯 Solución para el LookupError ---
# Forzar la descarga de los recursos necesarios de NLTK.
# Esto es CRUCIAL para Streamlit, ya que no tiene los recursos descargados.
def check_and_download_nltk_resources():
    """Verifica si los recursos de NLTK están disponibles y los descarga si es necesario."""
    resources = ['stopwords', 'punkt'] # Necesitas ambos para el preprocesamiento y stemming

    for resource in resources:
        try:
            # 2. Intenta encontrar el recurso
            nltk.data.find(f'tokenizers/{resource}') # Usa un path común para verificar
        except LookupError:
            # 3. Si no lo encuentra, lo descarga
            print(f"Descargando recurso de NLTK: {resource}")
            nltk.download(resource)

# Ejecutar la verificación y descarga al inicio del script
check_and_download_nltk_resources()
# ----------------------------------------

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer # Solo para definir tipos, si es necesario
from sklearn.linear_model import LogisticRegression # Solo para definir tipos, si es necesario

# --- ⚙️ Constantes y Herramientas de NLP ---
MODEL_PATH = 'sentiment_model_lr.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

# Ahora puedes definir las herramientas sin error:
english_stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ... [Resto de tu código: load_resources, limpiar_tweet_en_ingles, etc.] ...

# 1. Función para cargar los recursos eficientemente
@st.cache_resource
def load_resources():
    """Carga el vectorizador y el modelo solo una vez."""
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model

# Cargar recursos (se llama solo una vez)
vectorizer, model = load_resources() 

# 2. Definir las utilidades de Preprocesamiento (tu código anterior)
english_stopwords = set(stopwords.words('english'))
stemmer = PorterStemmer() 

def limpiar_tweet_en_ingles(texto):
    """
    Realiza la limpieza y tokenización de un tweet en inglés.
    """
    
    # 1. Limpieza de Ruido y Minúsculas
    texto = texto.lower()
    # Eliminar URLs, menciones y hashtags (mantener el texto del hashtag si es relevante,
    # pero para simplificar, eliminaremos el símbolo #)
    texto = re.sub(r'http\S+|www\S+|@\S+|#', '', texto) 
    # Eliminar puntuación y caracteres especiales (excepto espacios)
    texto = re.sub(r'[^\w\s]', '', texto) 
    
    # 2. Tokenización
    tokens = texto.split()
    
    # 3. Eliminar Stop Words
    # ¡Importante!: La palabra 'not' (no) se excluye de la eliminación para capturar la negación.
    tokens_limpios = [
        word for word in tokens 
        if word not in english_stopwords or word == 'not' # Conservamos 'not'
    ]
    
    # 4. Stemming (Opcional, pero recomendado para modelos ligeros)
    # Reduce 'running', 'ran' a 'run' para reducir el vocabulario
    tokens_stemmed = [stemmer.stem(word) for word in tokens_limpios]
    
    # 5. Reconstruir el texto limpio
    return " ".join(tokens_stemmed)

def predecir_sentimiento(texto_entrada):
    # 1. Limpieza
    texto_limpio = limpiar_tweet_en_ingles(texto_entrada)
    
    # 2. Vectorización
    # El vectorizador CARGADO transforma el texto
    X_input = vectorizer.transform([texto_limpio])
    
    # 3. Predicción
    # El modelo CARGADO realiza la predicción
    prediccion = model.predict(X_input)
    
    # 4. Interpretación del resultado (Asumiendo 0 y 4)
    if prediccion[0] == 4:
        return "Positivo 😊"
    else:
        return "Negativo 😔"
    
# Título de la Aplicación
st.title("Clasificador de Sentimiento de Tweets")
st.markdown("Utiliza Regresión Logística entrenada con el dataset Sentiment140.")

# Campo de entrada de texto para el usuario
user_input = st.text_area("Ingresa un Tweet en inglés aquí:", "I love machine learning and Streamlit!")

# Botón para clasificar
if st.button("Clasificar Sentimiento"):
    if user_input:
        # Llama a la función de predicción
        resultado = predecir_sentimiento(user_input)
        
        # Muestra el resultado
        st.subheader("Resultado de la Clasificación:")
        if "Positivo" in resultado:
            st.success(resultado) # Muestra en verde
        else:
            st.error(resultado) # Muestra en rojo
    else:
        st.warning("Por favor, ingresa un texto para clasificar.")
