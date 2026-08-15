from pathlib import Path
import json
import re
import time

import joblib
import nltk
import pandas as pd
import sklearn

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 0. Configuración
# ============================================================

RUTA_DATOS = Path("sentiment.csv")

RUTA_VECTORIZADOR = Path("tfidf_vectorizer_v2.pkl")
RUTA_MODELO = Path("sentiment_model_lr_v2.pkl")
RUTA_METRICAS = Path("metrics_v2.json")

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 1. Recursos NLTK
# ============================================================

nltk.download("stopwords", quiet=True)

english_stopwords = set(stopwords.words("english"))
stemmer = PorterStemmer()


# ============================================================
# 2. Preprocesamiento
# ============================================================

def limpiar_tweet(texto):
    """
    Replica el preprocesamiento usado en el proyecto original:
    - minúsculas
    - eliminación de URLs, menciones, hashtags y puntuación
    - eliminación de stopwords en inglés, conservando 'not'
    - stemming con PorterStemmer
    """

    texto = str(texto).lower()

    # Eliminar URLs, menciones y símbolo #
    texto = re.sub(r"http\S+|www\S+|@\S+|#", "", texto)

    # Eliminar puntuación y caracteres especiales
    texto = re.sub(r"[^\w\s]", "", texto)

    # Tokenización
    tokens = texto.split()

    # Stopwords, conservando "not"
    tokens_limpios = [
        palabra
        for palabra in tokens
        if palabra not in english_stopwords or palabra == "not"
    ]

    # Stemming
    tokens_stemmed = [
        stemmer.stem(palabra)
        for palabra in tokens_limpios
    ]

    return " ".join(tokens_stemmed)


# ============================================================
# 3. Carga de datos
# ============================================================

def cargar_sentiment140(ruta):
    """
    Lee Sentiment140 tanto si el CSV tiene encabezados
    como si viene en el formato original sin encabezados.
    """

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta.resolve()}"
        )

    print("Leyendo dataset...")

    df = pd.read_csv(ruta, encoding="latin1")

    if {"target", "text"}.issubset(df.columns):
        return df

    print(
        "No se detectaron encabezados. "
        "Leyendo con el formato original de Sentiment140..."
    )

    columnas = [
        "target",
        "ids",
        "date",
        "flag",
        "user",
        "text",
    ]

    df = pd.read_csv(
        ruta,
        encoding="latin1",
        header=None,
        names=columnas,
    )

    return df


# ============================================================
# 4. Carga y validación
# ============================================================

inicio = time.time()

df = cargar_sentiment140(RUTA_DATOS)

print(f"Observaciones cargadas: {len(df):,}")
print("\nDistribución original:")
print(df["target"].value_counts().sort_index())

# Nos quedamos únicamente con las clases esperadas
df = df[df["target"].isin([0, 4])].copy()

df = df.dropna(subset=["text", "target"])

print(f"\nObservaciones válidas: {len(df):,}")


# ============================================================
# 5. División train/test ANTES del TF-IDF
# ============================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"],
    df["target"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["target"],
)

print(f"\nEntrenamiento: {len(X_train_text):,}")
print(f"Prueba:         {len(X_test_text):,}")

print("\nDistribución en prueba:")
print(y_test.value_counts().sort_index())


# ============================================================
# 6. Preprocesamiento
# ============================================================

print("\nPreprocesando conjunto de entrenamiento...")
X_train_clean = X_train_text.apply(limpiar_tweet)

print("Preprocesando conjunto de prueba...")
X_test_clean = X_test_text.apply(limpiar_tweet)


# ============================================================
# 7. TF-IDF
# ============================================================

print("\nAjustando TF-IDF SOLO con entrenamiento...")

vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
)

X_train = vectorizer.fit_transform(X_train_clean)

print("Transformando conjunto de prueba...")
X_test = vectorizer.transform(X_test_clean)

print(f"Vocabulario final: {len(vectorizer.vocabulary_):,} características")


# ============================================================
# 8. Regresión logística
# ============================================================

print("\nEntrenando regresión logística...")

model_lr = LogisticRegression(
    max_iter=1000,
    solver="liblinear",
)

model_lr.fit(X_train, y_train)


# ============================================================
# 9. Evaluación
# ============================================================

print("\nEvaluando modelo...")

y_pred = model_lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision_pos = precision_score(
    y_test,
    y_pred,
    pos_label=4,
)

recall_pos = recall_score(
    y_test,
    y_pred,
    pos_label=4,
)

f1_pos = f1_score(
    y_test,
    y_pred,
    pos_label=4,
)

f1_macro = f1_score(
    y_test,
    y_pred,
    average="macro",
)

matriz = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 4],
)

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

print(f"Accuracy:             {accuracy:.4f}")
print(f"Precision positivo:   {precision_pos:.4f}")
print(f"Recall positivo:      {recall_pos:.4f}")
print(f"F1 positivo:          {f1_pos:.4f}")
print(f"F1 macro:             {f1_macro:.4f}")

print("\nMatriz de confusión [0, 4]:")
print(matriz)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
    )
)


# ============================================================
# 10. Guardar métricas
# ============================================================

metricas = {
    "dataset": "Sentiment140",
    "n_total": int(len(df)),
    "n_train": int(len(X_train_text)),
    "n_test": int(len(X_test_text)),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "accuracy": float(accuracy),
    "precision_positive": float(precision_pos),
    "recall_positive": float(recall_pos),
    "f1_positive": float(f1_pos),
    "f1_macro": float(f1_macro),
    "confusion_matrix": matriz.tolist(),
    "tfidf": {
        "max_features": 50000,
        "ngram_range": [1, 2],
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    },
    "logistic_regression": {
        "solver": "liblinear",
        "max_iter": 1000,
    },
    "sklearn_version": sklearn.__version__,
}

with open(RUTA_METRICAS, "w", encoding="utf-8") as archivo:
    json.dump(
        metricas,
        archivo,
        ensure_ascii=False,
        indent=4,
    )


# ============================================================
# 11. Guardar NUEVOS modelos
# ============================================================

joblib.dump(
    vectorizer,
    RUTA_VECTORIZADOR,
)

joblib.dump(
    model_lr,
    RUTA_MODELO,
)

print("\nArchivos guardados:")
print(f"- {RUTA_VECTORIZADOR}")
print(f"- {RUTA_MODELO}")
print(f"- {RUTA_METRICAS}")

duracion = time.time() - inicio

print(
    f"\nProceso terminado en "
    f"{duracion / 60:.1f} minutos."
)