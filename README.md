# 💬 Clasificador de Sentimiento de Tweets

Aplicación web para clasificar tweets como **positivos o negativos** mediante técnicas de procesamiento de lenguaje natural y aprendizaje automático.

El proyecto utiliza **TF-IDF** para representar el texto y un modelo de **Regresión Logística** entrenado sobre **1,6 millones de tweets de Sentiment140**. La aplicación fue desarrollada con **Streamlit** y permite analizar textos individuales en inglés o español, interpretar la influencia de palabras en la predicción y procesar archivos CSV.

## 📊 Resultados del modelo

El conjunto Sentiment140 se dividió de forma estratificada en **80 % entrenamiento y 20 % prueba**.

| Métrica                        |     Resultado |
| ------------------------------ | ------------: |
| Accuracy                       |   **79,93 %** |
| F1 macro                       |   **79,92 %** |
| Precision — clase positiva     |   **78,92 %** |
| Recall — clase positiva        |   **81,67 %** |
| F1 — clase positiva            |   **80,27 %** |
| Observaciones de entrenamiento | **1.280.000** |
| Observaciones de prueba        |   **320.000** |

El vectorizador TF-IDF se ajusta exclusivamente con el conjunto de entrenamiento para evitar fuga de información hacia el conjunto de prueba.

### Matriz de confusión

|                   | Predicho negativo | Predicho positivo |
| ----------------- | ----------------: | ----------------: |
| **Real negativo** |           125.099 |            34.901 |
| **Real positivo** |            29.335 |           130.665 |

Las métricas completas generadas durante el entrenamiento se encuentran en [`metrics.json`](metrics.json).

## ⚙️ Metodología

El flujo de modelamiento es:

```text
Sentiment140
1.600.000 tweets
        ↓
División estratificada 80 % / 20 %
        ↓
Preprocesamiento de texto
        ↓
TF-IDF
50.000 características
unigramas + bigramas
        ↓
Regresión Logística
        ↓
Evaluación sobre 320.000 tweets
```

### Preprocesamiento

Los textos son procesados mediante:

* Conversión a minúsculas.
* Eliminación de URLs, menciones, hashtags y puntuación.
* Eliminación de *stopwords* en inglés, conservando la negación `not`.
* *Stemming* mediante `PorterStemmer`.
* Vectorización TF-IDF con unigramas y bigramas.

## 🚀 Funcionalidades de la aplicación

La interfaz desarrollada en Streamlit permite:

* Clasificar textos como **positivos o negativos**.
* Mostrar las probabilidades estimadas por el modelo.
* Analizar textos en **inglés y español**.
* Traducir automáticamente textos en español al inglés antes de clasificarlos.
* Ajustar el umbral utilizado para clasificar un texto como positivo.
* Identificar palabras y bigramas con mayor influencia en la predicción.
* Visualizar las etapas de preprocesamiento aplicadas al texto.
* Procesar múltiples textos mediante archivos CSV.
* Descargar los resultados del análisis masivo.

> El modelo fue entrenado y evaluado sobre textos en inglés. El análisis de textos en español se realiza mediante una etapa previa de traducción automática y no representa una evaluación independiente del desempeño del modelo en español.

## 🛠️ Tecnologías

* Python
* pandas
* scikit-learn
* NLTK
* Streamlit
* deep-translator
* joblib
* Git y GitHub

## 📁 Estructura del proyecto

```text
sentiment_twitter_app/
│
├── app.py
├── train_model.py
├── metrics.json
├── requirements.txt
├── sentiment_model_lr.pkl
├── tfidf_vectorizer.pkl
├── .gitignore
└── README.md
```

### `app.py`

Contiene la aplicación web desarrollada con Streamlit y utiliza el modelo entrenado para realizar predicciones.

### `train_model.py`

Contiene el proceso reproducible de:

1. carga de Sentiment140;
2. división entrenamiento/prueba;
3. preprocesamiento;
4. ajuste del vectorizador TF-IDF;
5. entrenamiento de la regresión logística;
6. evaluación;
7. almacenamiento del modelo y las métricas.

### `metrics.json`

Guarda las métricas y parámetros principales obtenidos durante el entrenamiento.

## 💻 Ejecución local

### 1. Clonar el repositorio

```bash
git clone https://github.com/TAndres13/sentiment_twitter_app.git
cd sentiment_twitter_app
```

### 2. Crear un entorno virtual

En Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá en el navegador mediante el servidor local de Streamlit.

## 🔄 Reentrenamiento del modelo

El conjunto de datos no se incluye en el repositorio debido a su tamaño.

Para reproducir el entrenamiento:

1. Obtener el conjunto **Sentiment140**.
2. Guardar el archivo de entrenamiento en la raíz del proyecto con el nombre:

```text
sentiment.csv
```

3. Ejecutar:

```bash
python train_model.py
```

El script generará nuevamente:

```text
sentiment_model_lr.pkl
tfidf_vectorizer.pkl
metrics.json
```

## 📌 Alcance y limitaciones

* El problema se plantea como una clasificación binaria: sentimiento positivo o negativo.
* El modelo se entrenó sobre tweets en inglés.
* Los resultados reportados corresponden al conjunto de prueba de Sentiment140.
* El análisis en español depende de una traducción automática previa.
* Expresiones ambiguas, sarcasmo, ironía y lenguaje altamente contextual pueden ser difíciles de clasificar mediante un modelo basado en TF-IDF.

## 👥 Autores

Proyecto académico desarrollado por:

* Miguel Angel Hernandez Lizarazo
* Jesus Santiago Poveda
* Andres Felipe Torres Pinto
