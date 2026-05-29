import streamlit as st
from transformers import MBartForConditionalGeneration
from transformers import MBart50TokenizerFast
import torch

st.set_page_config(page_title="mBART Traductor")

st.title("🌍 Traductor Multilingüe con mBART")
st.write("Proyecto Final - Transformer Encoder Decoder")

# Modelo oficial del paper
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

@st.cache_resource
def cargar_modelo():
    tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
    model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = cargar_modelo()

# Idiomas
idiomas = {
    "Español": "es_XX",
    "Inglés": "en_XX",
    "Francés": "fr_XX",
    "Alemán": "de_DE",
    "Italiano": "it_IT",
    "Portugués": "pt_XX"
}

origen = st.selectbox("Idioma origen", list(idiomas.keys()))
destino = st.selectbox("Idioma destino", list(idiomas.keys()))

texto = st.text_area("Ingrese texto")

if st.button("Traducir"):

    tokenizer.src_lang = idiomas[origen]

    encoded = tokenizer(
        texto,
        return_tensors="pt"
    )

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[idiomas[destino]],
        max_length=40
    )

    traduccion = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0]

    st.success("Traducción")
    st.write(traduccion)