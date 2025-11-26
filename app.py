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

def predecir_sentimiento_con_confianza(texto_entrada):
    # 1. Limpieza y Vectorización (igual que antes)
    texto_limpio = limpiar_tweet_en_ingles(texto_entrada)
    X_input = vectorizer.transform([texto_limpio])

    # 2. Predecir Probabilidades
    # Devuelve un array con [Probabilidad de 0, Probabilidad de 4]
    # Ej: [[0.08, 0.92]]
    proba_array = model.predict_proba(X_input)[0] 

    # 3. Determinar el Sentimiento y la Confianza
    # Sentiment140 usa 0 (negativo) y 4 (positivo)

    if proba_array[1] > proba_array[0]:
        # Predicción positiva (la proba de 4 es mayor)
        confianza = proba_array[1] * 100
        etiqueta = "Positivo"
    else:
        # Predicción negativa (la proba de 0 es mayor o igual)
        confianza = proba_array[0] * 100
        etiqueta = "Negativo"

    return etiqueta, confianza # Retorna dos valores


# 1. Obtener los nombres de las características (palabras/bigramas) del vectorizador
feature_names = vectorizer.get_feature_names_out()

# 2. Obtener los coeficientes del modelo de Regresión Logística
# Nota: LogisticRegression.coef_[0] se usa para clasificación binaria
# Los coeficientes están en el mismo orden que 'feature_names'
coefficients = model.coef_[0]

# 3. Crear un diccionario de mapeo: {palabra: peso}
word_weights = dict(zip(feature_names, coefficients))

# 4. Definir un umbral (opcional, para filtrar palabras muy débiles)
# Puedes ajustar este valor
UMBRAL_PESO = 0.5

def analizar_palabras_clave(texto_entrada, word_weights, umbral):
    # 1. Limpieza y Tokenización del tweet de entrada
    # Reutilizamos la limpieza, pero solo necesitamos los tokens stemizados
    tokens_stemmed = limpiar_tweet_en_ingles(texto_entrada).split()
    
    palabras_encontradas = []
    
    # 2. Buscar tokens y bigramas en el diccionario de pesos
    
    # Búsqueda de Unigramas (Palabras solas)
    for token in tokens_stemmed:
        if token in word_weights and abs(word_weights[token]) >= umbral:
            palabras_encontradas.append((token, word_weights[token]))
            
    # Búsqueda de Bigramas (Pares de palabras):
    # Esto es crucial si usaste ngram_range=(1, 2) en el vectorizador.
    if len(tokens_stemmed) >= 2:
        for i in range(len(tokens_stemmed) - 1):
            bigrama = f"{tokens_stemmed[i]} {tokens_stemmed[i+1]}"
            if bigrama in word_weights and abs(word_weights[bigrama]) >= umbral:
                palabras_encontradas.append((bigrama, word_weights[bigrama]))

    # 3. Ordenar por peso (el más influyente primero)
    # Se ordena por el valor absoluto del peso para mostrar los más fuertes
    palabras_encontradas.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return palabras_encontradas[:5] # Devolver solo las 5 más influyentes

# Esta función la puedes llamar justo después de obtener la predicción.


#=================================
# Interefaz de Streamlit
#=================================


# Título de la Aplicación
st.title("Clasificador de Sentimiento de Tweets")
st.markdown("Utiliza Regresión Logística entrenada con el dataset Sentiment140.")

# Campo de entrada de texto para el usuario
user_input = st.text_area("Ingresa un Tweet en inglés aquí:", "I love machine learning and Streamlit!")

# Botón para clasificar
if st.button("Clasificar Sentimiento"):
    if user_input:

        # 1. Obtener la predicción y confianza (como lo hiciste antes)
        etiqueta, confianza = predecir_sentimiento_con_confianza(user_input) 
        
        # 2. Obtener las palabras clave del análisis
        top_palabras = analizar_palabras_clave(user_input, word_weights, UMBRAL_PESO)
        
        st.subheader("Resultado de la Clasificación:")
        # Muestra la confianza formateada a dos decimales
        st.info(f"Sentimiento: **{etiqueta}**")
        st.metric(label="Confianza del Modelo", value=f"{confianza:.2f}%")        
        # ... Muestra el resultado y la confianza ...
        
        st.divider()
        st.subheader("🔎 Contribución de Palabras Clave")
        
        if top_palabras:
            st.markdown("Las palabras más influyentes en esta clasificación son:")
            
            for palabra, peso in top_palabras:
                # Usar color basado en el peso (positivo o negativo)
                color = "green" if peso > 0 else "red"
                emoji = "⬆️" if peso > 0 else "⬇️"
                
                # Formatear el peso a 4 decimales
                peso_formateado = f"{peso:.4f}"
                
                st.markdown(
                    f"**:{color}[{palabra}]** ({emoji} Influencia: {peso_formateado})"
                )
        else:
            st.info("No se encontraron palabras clave con alta influencia para mostrar (pueden ser palabras neutras o raras).")


# En la sección de resultados del Streamlit:
st.divider()
with st.expander("Mostrar Análisis de Procesamiento del Modelo"):
    
    # a. Texto en minúsculas y limpieza inicial
    texto_minusculas = user_input.lower()
    st.markdown(f"**Paso 1: Normalización**")
    st.code(f"Texto en minúsculas y sin URLs/menciones: {texto_minusculas}")
    
    # b. El resultado final del preprocesamiento (texto limpio y stemizado)
    texto_stemizado = limpiar_tweet_en_ingles(user_input)
    st.markdown(f"**Paso 2: Stop Words y Stemming**")
    st.code(f"Resultado final enviado al vectorizador: {texto_stemizado}")
    
    # c. Análisis de longitud del texto
    st.markdown(f"**Paso 3: Métricas**")
    col1, col2 = st.columns(2)
    col1.metric("Longitud Original", f"{len(user_input)} caracteres")
    col2.metric("Tokens Limpios", f"{len(texto_stemizado.split())} palabras")
    
    # d. Mensaje final
    st.info("El modelo solo recibe el 'Resultado final' en formato numérico (TF-IDF).")