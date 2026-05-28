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

Los modelos tradicionales de traducción automática tienen dificultades para:
 - Traducir correctamente entre múltiples idiomas,
 - Mantener el contexto semántico,
 - Trabajar con idiomas con pocos datos
 - Generar traducciones coherentes.
Además, muchos sistemas requieren grandes cantidades de datos paralelos para cada idioma.
mBART busca resolver este problema mediante una arquitectura Transformer encoder–decoder multilingüe entrenada con denoising pre-training.

---

# Objetivos

General

Implementar y analizar la arquitectura Transformer encoder–decoder mBART para tareas de traducción automática multilingüe utilizando pesos preentrenados.


Específicos

 - Comprender el funcionamiento de la arquitectura mBART.
 - Implementar inferencia utilizando Hugging Face.
 - Explicar el funcionamiento del encoder y decoder.
 - Analizar el mecanismo de atención.
 - Evaluar traducciones multilingües.
 - Explicar el denoising pre-training.

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

---

# ¿Qué es mBART?

mBART es un modelo Transformer seq2seq (sequence-to-sequence) basado en la arquitectura BART.
Está diseñado para:
 - Traducción automática,
 - Generación de texto,
 - Tareas multilingües.
Utiliza:
 - Encoder,
 - Decoder,
 - Self-attention,
 - Cross-attention,
 - Embeddings multilingües.

---

# Funcionamiento General

## Flujo del modelo

Texto → Tokenización → Encoder → Attention → Decoder → Traducción

---

---

# Encoder

El encoder:
1. Recibe el texto de entrada,
2. Transforma palabras en embeddings,
3. Aplica self-attention,
4. Genera representaciones contextuales.

---

# Self-Attention

Permite que cada palabra analice su relación con las demás palabras de la oración para comprender el contexto.
Ejemplo:
"The cat sat on the mat"
La palabra "cat" puede relacionarse con "sat" y "mat" para entender el contexto.
---

---

# Decoder

El decoder:
1. Recibe la representación del encoder,
2. Utiliza cross-attention,
3. Genera texto token por token.
El decoder predice la siguiente palabra usando:
- palabras anteriores,
- contexto generado por el encoder.


---

# Cross-Attention

Permite que el decoder consulte la información generada por el encoder durante la generación de la traducción.

---

---

# Denoising Pre-training


mBART utiliza una estrategia llamada:
Denoising Pre-training
Proceso:
1. Se toma un texto original.
2. El texto se corrompe:
- se eliminan palabras,
- se desordenan oraciones,
- se ocultan fragmentos.
3. El modelo aprende a reconstruir el texto correcto.

Esto ayuda al modelo a aprender relaciones semánticas profundas.

---

# Q, K y V

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

---

# Conclusiones

mBART es una arquitectura Transformer encoder–decoder eficiente para traducción automática multilingüe.
El modelo utiliza:
 - self-attention,
 - cross-attention,
 - embeddings multilingües,
 - denoising pre-training.
 
La implementación usando Hugging Face permite realizar inferencia sin entrenar desde cero y comprender profundamente el funcionamiento de arquitecturas seq2seq modernas.


---

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
git clone https://github.com/LuisaOs/mBartTransformer.git