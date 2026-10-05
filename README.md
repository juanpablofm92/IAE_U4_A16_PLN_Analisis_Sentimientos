# Actividad 16: Analizador de Sentimientos con TF-IDF y Word2Vec

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 📌 1. Descripción del Proyecto

Este proyecto implementa una solución completa y modularizada de **Procesamiento de Lenguaje Natural (PLN)** en Python para el análisis de polaridad de sentimientos en opiniones y reseñas en idioma español. 

El sistema compara experimentalmente dos paradigmas de extracción de características:
1. **Representación Dispersa (Sparse):** TF-IDF con unigramas y bigramas vía `scikit-learn`.
2. **Representación Continua (Dense):** Embeddings de palabras entrenados vía `Word2Vec` (`gensim`).

Sobre ambas representaciones se evalúan tres arquitecturas de clasificación supervisada:
* Regresión Logística (`LogisticRegression`)
* Máquinas de Soporte Vectorial (`SVC` con kernel lineal)
* Bosques Aleatorios (`RandomForestClassifier`)

---

## 🛠️ 2. Arquitectura del Pipeline

```
Texto en Bruto
      │
      ▼
Normalización de Emoticonos (:), :D, ❤️, etc. -> tokens semánticos)
      │
      ▼
Normalización de Acentos y Caracteres (NFD preservando 'ñ')
      │
      ▼
Expansión y Fusión de Negaciones ('no bueno' -> 'no_bueno')
      │
      ▼
Filtrado Selectivo de Stopwords
      │
      ├───────────────────────────────┐
      ▼                               ▼
Extracción TF-IDF (1-2 n-grams)   Extracción Word2Vec (Embeddings)
      │                               │
      ├───────────────┬───────────────┤
      ▼               ▼               ▼
Logistic Reg.      Lin-SVM       Random Forest
      │               │               │
      └───────────────┼───────────────┘
                      ▼
     Evaluación (F1-Score, Matriz de Confusión)
```

---

## 🚀 3. Instrucciones de Instalación y Ejecución

### Requisitos previos
* Python 3.9 o superior.

### Instalación de dependencias
```bash
pip install -r requirements.txt
```

### Ejecución
```bash
python main.py
```

---

## 📊 4. Ejemplo de Salida en Consola

```text
===========================================================================
  TECNM / ITSU - MAESTRÍA EN IA - ACTIVIDAD 16 (PLN SENTIMIENTOS)
===========================================================================
[*] Dataset cargado: 30 muestras balanceadas.

[+] Demostración de Pipeline de Limpieza:
    Texto original: 'El servicio no es bueno y el paquete nunca llegó :( 😡'
    Texto limpio:   'servicio no_bueno paquete nunca_llego triste_emoji enojado_emoji'

--- 1. Extracción de Características: TF-IDF ---
>>> Clasificador: Regresión Logística
              precision    recall  f1-score   support
    Negativo      1.000     1.000     1.000         5
    Positivo      1.000     1.000     1.000         4
    accuracy                          1.000         9
```

---

## ⚖️ 5. Consideraciones Éticas en PLN

1. **Sesgo Dialectal:** Los sistemas de análisis de opiniones deben contemplar variedades léxicas regionales de Hispanoamérica para evitar penalizar arbitrariamente ciertas formas de expresión.
2. **Privacidad:** Aplicación de políticas estrictas de desidentificación de datos sensibles según normativas RGPD y LFPDPPP.
