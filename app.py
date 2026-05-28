import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# ==========================================
# TÍTULO
# ==========================================

st.title("🌎 Traductor Inglés → Español")

st.write("Proyecto Final - Transformer Encoder Decoder")

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():

    model_name = "Helsinki-NLP/opus-mt-en-es"

    tokenizer = MarianTokenizer.from_pretrained(model_name)

    model = MarianMTModel.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = cargar_modelo()

# ==========================================
# ENTRADA
# ==========================================

texto = st.text_area(
    "Ingrese texto en inglés",
    height=150
)

# ==========================================
# BOTÓN
# ==========================================

if st.button("Traducir"):

    if texto.strip() == "":
        st.warning("Ingrese un texto")

    else:

        inputs = tokenizer(
            texto,
            return_tensors="pt",
            padding=True
        )

        translated = model.generate(**inputs)

        resultado = tokenizer.decode(
            translated[0],
            skip_special_tokens=True
        )

        st.subheader("✅ Traducción")

        st.success(resultado)