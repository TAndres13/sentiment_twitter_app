import re
import joblib
import nltk
import pandas as pd
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer  # Solo para tipos
from sklearn.linear_model import LogisticRegression          # Solo para tipos

# ----------------------------------------
# Configuración de página
# ----------------------------------------
st.set_page_config(
    page_title="Clasificador de Sentimiento de Tweets",
    page_icon="💬",
    layout="centered",
)

# ----------------------------------------
# Descarga y verificación de recursos NLTK
# ----------------------------------------
def check_and_download_nltk_resources():
    """
    Verifica si los recursos de NLTK están disponibles y los descarga si falta alguno.
    """
    resources = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
    }

    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Descargando recurso de NLTK: {resource}")
            nltk.download(resource)


check_and_download_nltk_resources()

# ----------------------------------------
# Constantes y herramientas NLP
# ----------------------------------------
MODEL_PATH = "sentiment_model_lr.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

english_stopwords = set(stopwords.words("english"))
stemmer = PorterStemmer()


# ----------------------------------------
# Carga de recursos con cache
# ----------------------------------------
@st.cache_resource
def load_resources():
    """
    Carga el vectorizador, el modelo y calcula los pesos por palabra
    solo una vez (se cachea entre recargas).
    """
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    # Palabras (features) y coeficientes del modelo
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    word_weights = dict(zip(feature_names, coefficients))

    return vectorizer, model, word_weights


vectorizer, model, word_weights = load_resources()


# ----------------------------------------
# Funciones de preprocesamiento
# ----------------------------------------
def limpiar_tweet_en_ingles(texto: str) -> str:
    """
    Limpieza y normalización de un tweet en inglés:
    - Minúsculas
    - Eliminación de URLs, menciones, hashtags, puntuación
    - Eliminación de stopwords (salvando 'not')
    - Stemming
    """
    # 1. Minúsculas
    texto = texto.lower()

    # 2. Eliminar URLs, menciones y hashtags
    texto = re.sub(r"http\S+|www\S+|@\S+|#", "", texto)

    # 3. Eliminar puntuación y caracteres especiales
    texto = re.sub(r"[^\w\s]", "", texto)

    # 4. Tokenización simple
    tokens = texto.split()

    # 5. Eliminar stopwords, pero conservar 'not' para negaciones
    tokens_limpios = [
        word for word in tokens
        if (word not in english_stopwords) or (word == "not")
    ]

    # 6. Stemming
    tokens_stemmed = [stemmer.stem(word) for word in tokens_limpios]

    # 7. Reconstruir
    return " ".join(tokens_stemmed)


def predecir_sentimiento_con_confianza(texto_entrada: str, umbral: float = 0.5):
    """
    Predice el sentimiento de un texto:
    - Devuelve etiqueta ("Positivo"/"Negativo")
    - Confianza (porcentaje)
    - Vector de probabilidades (negativo, positivo)
    """
    texto_limpio = limpiar_tweet_en_ingles(texto_entrada)
    X_input = vectorizer.transform([texto_limpio])

    proba_array = model.predict_proba(X_input)[0]  # [p0, p4]

    # Clasificación según umbral de positividad
    if proba_array[1] >= umbral:
        etiqueta = "Positivo"
        confianza = proba_array[1] * 100
    else:
        etiqueta = "Negativo"
        confianza = proba_array[0] * 100

    return etiqueta, confianza, proba_array


def analizar_palabras_clave(texto_entrada: str,
                            word_weights: dict,
                            umbral_peso: float = 0.5,
                            max_palabras: int = 5):
    """
    Analiza qué tokens (unigramas y bigramas) del texto
    tienen mayor peso en el modelo, según word_weights y un umbral.
    """
    tokens_stemmed = limpiar_tweet_en_ingles(texto_entrada).split()
    palabras_encontradas = []

    # Unigramas
    for token in tokens_stemmed:
        if token in word_weights and abs(word_weights[token]) >= umbral_peso:
            palabras_encontradas.append((token, word_weights[token]))

    # Bigramas (si el vectorizador los tiene)
    if len(tokens_stemmed) >= 2:
        for i in range(len(tokens_stemmed) - 1):
            bigrama = f"{tokens_stemmed[i]} {tokens_stemmed[i+1]}"
            if bigrama in word_weights and abs(word_weights[bigrama]) >= umbral_peso:
                palabras_encontradas.append((bigrama, word_weights[bigrama]))

    # Ordenar por influencia absoluta
    palabras_encontradas.sort(key=lambda x: abs(x[1]), reverse=True)

    return palabras_encontradas[:max_palabras]


# ----------------------------------------
# Sidebar: información y controles
# ----------------------------------------
st.sidebar.title("ℹ️ Sobre la app")
st.sidebar.write(
    """
Esta aplicación clasifica el **sentimiento** de tweets en inglés
usando **Regresión Logística** entrenada con el dataset *Sentiment140*.
"""
)
st.sidebar.markdown("**Autores:")

st.sidebar.markdown("---")
st.sidebar.markdown("### Ajustes del modelo")

umbral_positivo = st.sidebar.slider(
    "Umbral para clasificar como positivo",
    min_value=0.5,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Si la probabilidad de positivo supera este valor, se clasifica como 'Positivo'.",
)

umbral_peso_palabras = st.sidebar.slider(
    "Umbral de influencia para palabras clave",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="Filtra palabras/bigramas con pesos muy pequeños en el modelo.",
)


# ----------------------------------------
# Interfaz principal
# ----------------------------------------
st.title("💬 Clasificador de Sentimiento de Tweets")
st.markdown(
    """
Esta herramienta analiza el sentimiento de un tweet en inglés
y muestra **qué palabras tuvieron más peso** en la decisión del modelo.
"""
)

# Entrada de texto
user_input = st.text_area(
    "Ingresa un Tweet en inglés:",
    "I love machine learning and Streamlit!",
    height=100,
)

# Botón para clasificar
if st.button("Clasificar Sentimiento"):
    if user_input.strip():
        etiqueta, confianza, proba_array = predecir_sentimiento_con_confianza(
            user_input,
            umbral=umbral_positivo,
        )
        top_palabras = analizar_palabras_clave(
            user_input,
            word_weights,
            umbral_peso=umbral_peso_palabras,
        )

        # Resultado principal
        st.subheader("Resultado de la clasificación")
        st.info(f"Sentimiento: **{etiqueta}**")
        st.metric(label="Confianza del modelo", value=f"{confianza:.2f}%")

        # Probabilidades como gráfico
        st.markdown("#### Distribución de probabilidades")
        df_proba = pd.DataFrame(
            {
                "Sentiment": ["Negativo (0)", "Positivo (4)"],
                "Probabilidad": proba_array,
            }
        ).set_index("Sentiment")
        st.bar_chart(df_proba)

        st.divider()
        st.subheader("🔎 Contribución de palabras clave")

        if top_palabras:
            st.markdown("Palabras o bigramas más influyentes según el modelo:")

            # Tabla
            df_palabras = pd.DataFrame(top_palabras, columns=["Token", "Peso"])
            st.table(df_palabras)

            # Versión coloreada
            for palabra, peso in top_palabras:
                color = "green" if peso > 0 else "red"
                emoji = "⬆️" if peso > 0 else "⬇️"
                peso_formateado = f"{peso:.4f}"

                st.markdown(
                    f"**:{color}[{palabra}]** "
                    f"({emoji} influencia: `{peso_formateado}`)"
                )
        else:
            st.info(
                "No se encontraron palabras clave con alta influencia. "
                "Puede que el tweet tenga palabras muy neutras o poco frecuentes."
            )


# ----------------------------------------
# Expander: análisis del procesamiento del modelo
# ----------------------------------------
st.divider()
with st.expander("Mostrar análisis del preprocesamiento del texto"):
    st.markdown("### Paso 1: Normalización básica")
    texto_minusculas = user_input.lower()
    st.code(f"Texto en minúsculas:\n{texto_minusculas}")

    st.markdown("### Paso 2: Limpieza de ruido (URLs, menciones, hashtags, puntuación)")
    texto_sin_ruido = re.sub(r"http\S+|www\S+|@\S+|#", "", texto_minusculas)
    texto_sin_ruido = re.sub(r"[^\w\s]", "", texto_sin_ruido)
    st.code(f"Texto sin URLs/menciones/hashtags/puntuación:\n{texto_sin_ruido}")

    st.markdown("### Paso 3: Stopwords y stemming")
    texto_stemizado = limpiar_tweet_en_ingles(user_input)
    st.code(f"Texto final enviado al vectorizador (TF-IDF):\n{texto_stemizado}")

    st.markdown("### Paso 4: Métricas del texto")
    col1, col2 = st.columns(2)
    col1.metric("Longitud original", f"{len(user_input)} caracteres")
    col2.metric("Tokens finales", f"{len(texto_stemizado.split())} palabras")

    st.info("El modelo recibe este texto procesado y lo convierte en un vector TF-IDF.")


# ----------------------------------------
# Análisis masivo por CSV
# ----------------------------------------
st.divider()
st.header("📂 Análisis masivo de tweets (CSV)")

st.markdown(
    """
Sube un archivo **CSV** que contenga una columna llamada `text`
con los tweets que quieres analizar.
"""
)

uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("El CSV debe tener una columna llamada `text`.")
        else:
            # Preprocesamiento
            df["texto_limpio"] = df["text"].astype(str).apply(limpiar_tweet_en_ingles)
            X = vectorizer.transform(df["texto_limpio"])
            proba = model.predict_proba(X)

            # Clasificación según umbral global
            df["proba_neg"] = proba[:, 0]
            df["proba_pos"] = proba[:, 1]
            df["sentiment_pred"] = df["proba_pos"].apply(
                lambda p: "Positivo" if p >= umbral_positivo else "Negativo"
            )

            st.markdown("### Vista previa de resultados")
            st.dataframe(df.head())

            # Botón de descarga
            csv_resultado = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar resultados como CSV",
                data=csv_resultado,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Ocurrió un error al leer el archivo: {e}")


# ----------------------------------------
# Detalles técnicos del modelo
# ----------------------------------------
with st.expander("Detalles técnicos del modelo"):
    st.markdown(
        """
- **Modelo:** Regresión Logística binaria (`0 = negativo`, `4 = positivo`).
- **Representación del texto:** TF–IDF (unigramas y posiblemente bigramas).
- **Dataset de entrenamiento:** [Sentiment140](http://help.sentiment140.com/for-students)
  con tweets etiquetados automáticamente.
- **Preprocesamiento:**
  - Minúsculas  
  - Eliminación de URLs, menciones, hashtags y puntuación  
  - Stopwords en inglés (salvando `not`)  
  - Stemming con `PorterStemmer`  
- Cada coeficiente del modelo indica **qué tanto** una palabra o bigrama
  empuja la predicción hacia lo positivo o lo negativo.
"""
    )
