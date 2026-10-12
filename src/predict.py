"""
Predicción de sentimiento con el modelo ya entrenado.

Uso desde la terminal:

    python src/predict.py "This movie was amazing, I loved every minute"

También lo usa la aplicación de Streamlit (app.py), así que la limpieza y la
codificación de un texto nuevo viven en un solo lugar.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from tensorflow import keras

RAIZ = Path(__file__).resolve().parent.parent
RUTA_MODELO = RAIZ / "models" / "lstm_imdb.keras"
RUTA_VOCABULARIO = RAIZ / "models" / "vocabulario.json"
RUTA_CONFIG = RAIZ / "models" / "config.json"

# Índices reservados por Keras en el dataset IMDB
PAD, INICIO, DESCONOCIDA = 0, 1, 2


def limpiar_texto(texto):
    """Minúsculas, sin etiquetas HTML y sin signos de puntuación (se conservan los apóstrofos)."""
    texto = texto.lower()
    texto = re.sub(r"<br\s*/?>", " ", texto)
    texto = re.sub(r"[^a-z0-9'\s]", " ", texto)
    return texto.split()


def codificar(texto, vocabulario, longitud):
    """Convierte un texto en la secuencia de números que espera la red, con la misma forma que el entrenamiento."""
    palabras = limpiar_texto(texto)
    secuencia = [INICIO] + [vocabulario.get(p, DESCONOCIDA) for p in palabras]
    desconocidas = sum(1 for p in palabras if p not in vocabulario)
    matriz = keras.utils.pad_sequences([secuencia], maxlen=longitud)
    return matriz, len(palabras), desconocidas


def cargar_modelo():
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {RUTA_MODELO}. "
            "Entrénalo primero con el notebook o con src/train.py.")
    modelo = keras.models.load_model(RUTA_MODELO)
    vocabulario = json.loads(RUTA_VOCABULARIO.read_text(encoding="utf-8"))
    config = json.loads(RUTA_CONFIG.read_text(encoding="utf-8"))
    return modelo, vocabulario, config


def predecir(modelo, vocabulario, config, texto):
    """Devuelve la clase predicha, la confianza, la probabilidad de 'positiva' y datos del texto."""
    matriz, n_palabras, desconocidas = codificar(texto, vocabulario, config["longitud"])
    prob_positiva = float(modelo.predict(matriz, verbose=0)[0][0])
    if prob_positiva >= 0.5:
        clase, confianza = config["clases"][1], prob_positiva
    else:
        clase, confianza = config["clases"][0], 1 - prob_positiva
    return {
        "clase": clase,
        "confianza": confianza,
        "prob_positiva": prob_positiva,
        "palabras": n_palabras,
        "desconocidas": desconocidas,
    }


def main():
    if len(sys.argv) < 2:
        print('Uso: python src/predict.py "texto de la reseña en inglés"')
        sys.exit(1)

    modelo, vocabulario, config = cargar_modelo()
    r = predecir(modelo, vocabulario, config, " ".join(sys.argv[1:]))
    print(f"Sentimiento: {r['clase']} ({r['confianza']:.1%})")
    print(f"Probabilidad de positiva: {r['prob_positiva']:.3f}")
    print(f"Palabras: {r['palabras']} | Fuera del vocabulario: {r['desconocidas']}")


if __name__ == "__main__":
    main()
