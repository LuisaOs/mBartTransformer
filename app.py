import streamlit as st
from transformers import MBartForConditionalGeneration
from transformers import MBart50TokenizerFast

st.title("Traductor Multilingüe con mBART")

model_name = "facebook/mbart-large-50-many-to-many-mmt"

model = MBartForConditionalGeneration.from_pretrained(model_name)

tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

texto = st.text_area("Ingrese texto en inglés")

if st.button("Traducir"):

    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "es_XX"

    inputs = tokenizer(texto, return_tensors="pt")

    translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.convert_tokens_to_ids("es_XX"),
    max_length=60,
    num_beams=4,
    early_stopping=True
)

    traduccion = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )

    st.write(traduccion[0])