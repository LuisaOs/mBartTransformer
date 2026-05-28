# mBART Transformer Project

## Proyecto Final — Procesamiento de Datos Secuenciales

Implementación de una arquitectura Transformer Encoder–Decoder basada en mBART para traducción automática multilingüe utilizando pesos preentrenados e inferencia funcional.

---

# Integrantes

- Manuel Castillo
- Luis Hurtado
- Luisa Ospina
- Juan Orozco

---

# Artículo Científico Base

## Multilingual Denoising Pre-training for Neural Machine Translation

## Link del Paper
https://huggingface.co/papers/2001.08210

## Modelo Utilizado
https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt

---

# Objetivo

Implementar y analizar una arquitectura Transformer encoder–decoder aplicada a traducción automática multilingüe utilizando el modelo mBART y pesos preentrenados.

---

# Problemática

Los sistemas tradicionales de traducción automática presentan dificultades para:

- trabajar con múltiples idiomas,
- mantener coherencia semántica,
- traducir idiomas con pocos recursos,
- y generar traducciones contextualmente correctas.

mBART propone una arquitectura Transformer multilingüe basada en denoising pre-training para mejorar traducción automática y transferencia entre idiomas.

---

# Arquitectura Utilizada

El proyecto utiliza mBART, una arquitectura:

- Transformer
- Encoder–Decoder
- Sequence-to-Sequence (Seq2Seq)

El modelo utiliza:

- Self-Attention
- Cross-Attention
- Embeddings Multilingües
- Denoising Pre-training

---

# Funcionamiento General

## Flujo del modelo

Texto → Tokenización → Encoder → Attention → Decoder → Traducción

---

# Self-Attention

Permite que cada palabra analice su relación con las demás palabras de la oración para comprender el contexto.

---

# Cross-Attention

Permite que el decoder consulte la información generada por el encoder durante la generación de la traducción.

---

# 🔍 Q, K y V

Los tensores de atención se generan mediante:

Q = XWQ  
K = XWK  
V = XWV

Donde:

- Q (Query): consulta información
- K (Key): representa información disponible
- V (Value): contiene la información contextual

---

# Fórmula de Attention

Attention(Q,K,V)=softmax((QK^T)/sqrt(d_k))V

---

# Innovaciones de mBART

- Traducción multilingüe
- Denoising pre-training
- Transferencia entre idiomas
- Mejor desempeño en idiomas de bajos recursos

---

# Tecnologías Utilizadas

- Python
- Hugging Face Transformers
- PyTorch
- Streamlit

---

# Instalación

## Clonar repositorio

```bash
git clone