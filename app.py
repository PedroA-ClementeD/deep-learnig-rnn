"""
Aplicación Streamlit: escribir una reseña de película y obtener su sentimiento con la LSTM entrenada.

Ejecutar desde la raíz del repositorio:

    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from src.predict import cargar_modelo, predecir  # noqa: E402

st.set_page_config(page_title="Sentimiento de reseñas", page_icon="🎬")

EJEMPLOS = {
    "Escribir mi propia reseña": "",
    "Ejemplo positivo": "One of the best films I have seen in years. The acting is brilliant "
                        "and the story kept me hooked until the very last scene.",
    "Ejemplo negativo": "A complete waste of time. The plot makes no sense, the dialogue is "
                        "awful and I almost fell asleep halfway through.",
    "Ejemplo mixto": "The visuals are beautiful, but the story is slow and the ending "
                     "left me disappointed.",
}


@st.cache_resource
def obtener_modelo():
    # Se carga una sola vez, no en cada interacción con la página
    return cargar_modelo()


st.title("Análisis de sentimiento de reseñas")
st.write(
    "Escribe una reseña de una película **en inglés** y la red LSTM dirá si es positiva "
    "o negativa, con su nivel de confianza.")
st.caption(
    "El modelo se entrenó con 20.000 reseñas de IMDB y conoce las 10.000 palabras más "
    "frecuentes. Si escribes en español casi todas las palabras le resultarán desconocidas.")

try:
    modelo, vocabulario, config = obtener_modelo()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

ejemplo = st.selectbox("Puedes partir de un ejemplo:", list(EJEMPLOS.keys()))
texto = st.text_area("Reseña", value=EJEMPLOS[ejemplo], height=160,
                     placeholder="Write your movie review here...")

if st.button("Analizar sentimiento", type="primary"):
    if not texto.strip():
        st.warning("Escribe una reseña primero.")
        st.stop()

    r = predecir(modelo, vocabulario, config, texto)

    if r["palabras"] == 0:
        st.warning("El texto no tiene palabras que se puedan analizar.")
        st.stop()

    icono = "😊" if r["clase"] == "positiva" else "😞"
    st.subheader(f"{icono} Sentimiento: {r['clase']}")
    st.metric("Confianza", f"{r['confianza']:.1%}")
    st.progress(r["prob_positiva"], text=f"Probabilidad de positiva: {r['prob_positiva']:.1%}")

    if r["confianza"] < 0.6:
        st.info("La confianza es baja: la reseña parece tener una mezcla de opiniones.")

    proporcion = r["desconocidas"] / r["palabras"]
    st.caption(f"Palabras analizadas: {r['palabras']} · Fuera del vocabulario: "
               f"{r['desconocidas']} ({proporcion:.0%})")
    if proporcion > 0.5:
        st.warning("Más de la mitad de las palabras no están en el vocabulario del modelo. "
                   "¿Está escrita en inglés?")
