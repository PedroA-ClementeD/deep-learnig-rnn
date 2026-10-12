"""
Entrenamiento de la LSTM sobre el dataset IMDB.

Es la versión en script de lo que hace el notebook. Sirve para entrenar sin abrir Colab:

    python src/train.py
    python src/train.py --epocas 5

Deja el modelo, el vocabulario y la configuración en models/, y las gráficas y
métricas en reports/.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # permite guardar gráficas sin pantalla
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_MODELOS = RAIZ / "models"
CARPETA_REPORTES = RAIZ / "reports"

NUM_PALABRAS = 10000   # tamaño del vocabulario
LONGITUD = 200         # palabras por reseña después de recortar o rellenar
CLASES = ["negativa", "positiva"]
SEMILLA = 42


def cargar_datos():
    """Carga IMDB, rellena las secuencias y separa entrenamiento, validación y prueba."""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=NUM_PALABRAS)

    x_train_full = keras.utils.pad_sequences(x_train_full, maxlen=LONGITUD)
    x_test = keras.utils.pad_sequences(x_test, maxlen=LONGITUD)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.20,
        stratify=y_train_full, random_state=SEMILLA)

    return x_train, y_train, x_val, y_val, x_test, y_test


def construir_vocabulario():
    """Diccionario palabra -> número, solo con las palabras que conoce el modelo."""
    indice = keras.datasets.imdb.get_word_index()
    return {palabra: i + 3 for palabra, i in indice.items() if i + 3 < NUM_PALABRAS}


def construir_modelo():
    modelo = keras.Sequential([
        keras.Input(shape=(LONGITUD,)),
        layers.Embedding(input_dim=NUM_PALABRAS, output_dim=64),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.5),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="lstm_imdb")
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
    return modelo


def graficar_entrenamiento(historial, ruta):
    h = historial.history
    epocas = range(1, len(h["loss"]) + 1)
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4))
    ejes[0].plot(epocas, h["accuracy"], marker="o", label="Entrenamiento")
    ejes[0].plot(epocas, h["val_accuracy"], marker="o", label="Validación")
    ejes[0].set_title("Accuracy por época")
    ejes[0].set_xlabel("Época")
    ejes[0].legend()
    ejes[1].plot(epocas, h["loss"], marker="o", label="Entrenamiento")
    ejes[1].plot(epocas, h["val_loss"], marker="o", label="Validación")
    ejes[1].set_title("Pérdida (loss) por época")
    ejes[1].set_xlabel("Época")
    ejes[1].legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def graficar_matriz(y_real, y_pred, ruta):
    matriz = confusion_matrix(y_real, y_pred)
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASES, yticklabels=CLASES)
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase real")
    plt.title("Matriz de confusión (conjunto de prueba)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Entrena la LSTM sobre IMDB")
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    keras.utils.set_random_seed(SEMILLA)
    CARPETA_MODELOS.mkdir(exist_ok=True)
    CARPETA_REPORTES.mkdir(exist_ok=True)

    x_train, y_train, x_val, y_val, x_test, y_test = cargar_datos()
    print(f"Entrenamiento: {len(x_train)} | Validación: {len(x_val)} | Prueba: {len(x_test)}")

    modelo = construir_modelo()
    modelo.summary()

    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                               restore_best_weights=True)]
    historial = modelo.fit(x_train, y_train,
                           validation_data=(x_val, y_val),
                           epochs=args.epocas, batch_size=args.batch,
                           callbacks=callbacks, verbose=2)

    perdida, accuracy = modelo.evaluate(x_test, y_test, verbose=0)
    probabilidades = modelo.predict(x_test, verbose=0).ravel()
    y_pred = (probabilidades >= 0.5).astype(int)
    auc = roc_auc_score(y_test, probabilidades)
    reporte = classification_report(y_test, y_pred, labels=[0, 1], target_names=CLASES,
                                    digits=3, zero_division=0)
    print(f"\nAccuracy en prueba: {accuracy:.4f} | Pérdida: {perdida:.4f} | AUC: {auc:.4f}\n")
    print(reporte)

    graficar_entrenamiento(historial, CARPETA_REPORTES / "entrenamiento.png")
    graficar_matriz(y_test, y_pred, CARPETA_REPORTES / "matriz_confusion.png")
    (CARPETA_REPORTES / "metricas.txt").write_text(
        f"Accuracy prueba: {accuracy:.4f}\nPérdida prueba: {perdida:.4f}\nAUC: {auc:.4f}\n\n{reporte}",
        encoding="utf-8")

    modelo.save(CARPETA_MODELOS / "lstm_imdb.keras")
    (CARPETA_MODELOS / "vocabulario.json").write_text(
        json.dumps(construir_vocabulario(), ensure_ascii=False), encoding="utf-8")
    (CARPETA_MODELOS / "config.json").write_text(
        json.dumps({"num_palabras": NUM_PALABRAS, "longitud": LONGITUD, "clases": CLASES},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("Modelo, vocabulario y configuración guardados en models/")


if __name__ == "__main__":
    main()
