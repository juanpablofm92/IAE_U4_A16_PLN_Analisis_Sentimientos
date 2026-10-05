"""
=============================================================================
TECNOLÓGICO NACIONAL DE MÉXICO - INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN
DIVISIÓN DE ESTUDIOS DE POSGRADO E INVESTIGACIÓN
MAESTRÍA EN INTELIGENCIA ARTIFICIAL

Materia: Inteligencia Artificial y su Ética
Actividad 16: PLN - Analizador de Sentimientos con TF-IDF y Word2Vec
Alumno: Juan Pablo Figueroa Moran (Matrícula: M26040059)
=============================================================================
"""

import sys
import re
import unicodedata
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# 1. DICCIONARIOS DE PREPROCESAMIENTO Y EMOTICONOS
# =====================================================================

EMOTICON_DICT = {
    r":\)": " feliz_emoji ",
    r":-\)": " feliz_emoji ",
    r":D": " risa_emoji ",
    r":-D": " risa_emoji ",
    r":\(": " triste_emoji ",
    r":-\(": " triste_emoji ",
    r":'": " llorar_emoji ",
    r";\)": " complice_emoji ",
    r"<3": " amor_emoji ",
    r"❤️": " amor_emoji ",
    r"💔": " desamor_emoji ",
    r"😡": " enojado_emoji ",
    r"😠": " enojado_emoji ",
    r"😊": " sonriente_emoji ",
    r"😢": " llorar_emoji ",
    r"👍": " pulgar_arriba_emoji ",
    r"👎": " pulgar_abajo_emoji ",
    r"⭐": " estrella_emoji ",
    r"🔥": " fuego_emoji ",
    r"💩": " basura_emoji ",
}

NEGATION_WORDS = {"no", "nunca", "jamas", "jamás", "tampoco", "nadie", "nada", "ni", "sin"}

SPANISH_STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
    "para", "con", "una", "su", "al", "lo", "como", "mas", "pero", "sus", "le", "ya",
    "o", "este", "si", "porque", "esta", "entre", "cuando", "muy", "sobre", "tambien",
    "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto",
    "mi", "antes", "algunos", "unos", "yo", "otro", "otras", "otra", "el", "tanto", "esa"
}

# Remover negaciones de stopwords para preservar la semántica del sentimiento
SPANISH_STOPWORDS = SPANISH_STOPWORDS - NEGATION_WORDS


# =====================================================================
# 2. PIPELINE DE LIMPIEZA DE TEXTO EN ESPAÑOL Y EXPANSIÓN DE NEGACIONES
# =====================================================================

def normalizar_emoticonos(texto: str) -> str:
    """Mapea emoticonos gráficos y caracteres ASCII a tokens semánticos."""
    for pattern, replacement in EMOTICON_DICT.items():
        texto = re.sub(pattern, replacement, texto)
    return texto

def expandir_negaciones(texto: str) -> str:
    """
    Expande negaciones uniendo la palabra de negación con la palabra contigua.
    Ejemplo: 'no bueno' -> 'no_bueno', 'nunca sirve' -> 'nunca_sirve'.
    Esto previene que 'no' y 'bueno' se interpreten como tokens independientes.
    """
    palabras = texto.split()
    resultado = []
    i = 0
    while i < len(palabras):
        palabra = palabras[i]
        if palabra in NEGATION_WORDS and (i + 1) < len(palabras):
            siguiente = palabras[i + 1]
            resultado.append(f"{palabra}_{siguiente}")
            i += 2
        else:
            resultado.append(palabra)
            i += 1
    return " ".join(resultado)

def limpiar_texto_espanol(texto: str) -> str:
    """
    Pipeline integral de limpieza:
    1. Conversión a minúsculas
    2. Manejo y preservación semántica de emoticonos
    3. Normalización de caracteres y acentos (preservando la 'ñ')
    4. Expansión de secuencias de negación
    5. Eliminación de stopwords irrelevantes
    """
    texto = texto.lower()
    texto = normalizar_emoticonos(texto)
    
    # Preservar la 'ñ' temporalmente
    texto = texto.replace("ñ", "__enie__")
    texto = unicodedata.normalize('NFD', texto)
    texto = "".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    texto = texto.replace("__enie__", "ñ")
    
    # Remover caracteres especiales y dígitos excepto guiones bajos para tokens compuestos
    texto = re.sub(r"[^a-záéíóúñ\s_]", " ", texto)
    texto = expandir_negaciones(texto)
    
    # Filtrar tokens cortos y stopwords
    tokens = [tok for tok in texto.split() if tok not in SPANISH_STOPWORDS and len(tok) > 1]
    return " ".join(tokens)


# =====================================================================
# 3. MÓDULO DE EXTRACCIÓN DE CARACTERÍSTICAS (TF-IDF vs WORD2VEC)
# =====================================================================

class ExtractorWord2Vec:
    """
    Extractor de embeddings continuos promediados usando Word2Vec de gensim.
    Si gensim no está disponible, utiliza una aproximación vectorial continua determinista.
    """
    def __init__(self, vector_size: int = 50, window: int = 5, min_count: int = 1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        tokenized_corpus = [doc.split() for doc in corpus]
        try:
            from gensim.models import Word2Vec
            self.model = Word2Vec(sentences=tokenized_corpus,
                                  vector_size=self.vector_size,
                                  window=self.window,
                                  min_count=self.min_count,
                                  workers=2,
                                  seed=42)
            return self.transform(corpus)
        except Exception:
            # Fallback robusto continuo con hash embedding
            return self._hash_embeddings(corpus)

    def transform(self, corpus: List[str]) -> np.ndarray:
        if self.model is None:
            return self._hash_embeddings(corpus)
        
        vectors = []
        for doc in corpus:
            tokens = doc.split()
            valid_vecs = [self.model.wv[word] for word in tokens if word in self.model.wv]
            if len(valid_vecs) > 0:
                vectors.append(np.mean(valid_vecs, axis=0))
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.array(vectors)

    def _hash_embeddings(self, corpus: List[str]) -> np.ndarray:
        vectors = []
        for doc in corpus:
            vec = np.zeros(self.vector_size)
            tokens = doc.split()
            for t in tokens:
                np.random.seed(abs(hash(t)) % (2**31 - 1))
                vec += np.random.normal(0, 1, self.vector_size)
            if len(tokens) > 0:
                vec /= len(tokens)
            vectors.append(vec)
        return np.array(vectors)


# =====================================================================
# 4. CONJUNTO DE DATOS SINTÉTICO/CONTROLADO EN ESPAÑOL
# =====================================================================

def cargar_dataset_sentimientos() -> Tuple[List[str], List[int]]:
    """
    Dataset balanceado de 30 reseñas en español sobre productos y servicios
    (1: Positivo, 0: Negativo), diseñadas para poner a prueba negaciones y emoticonos.
    """
    data = [
        # Positivos
        ("Excelente servicio, llegó súper rápido y en perfecto estado :) ❤️", 1),
        ("Me encantó la calidad del producto, totalmente recomendado 👍", 1),
        ("La atención al cliente fue maravillosa, resolvieron todo de inmediato.", 1),
        ("No esperaba que fuera tan increíble, superó mis expectativas!", 1),
        ("Una experiencia fascinante, volveré a comprar sin duda alguna :D", 1),
        ("Todo excelente, empaque impecable y producto auténtico ⭐", 1),
        ("Muy buena atención, el personal fue muy atento y amable.", 1),
        ("Funciona de maravilla, no puedo pedir nada mejor ❤️", 1),
        ("Gran calidad precio, una de las mejores inversiones que he hecho.", 1),
        ("Rápido, confiable y seguro. 10 de 10 😊", 1),
        ("No tengo ninguna queja, el soporte fue fenomenal.", 1),
        ("El dispositivo responde muy bien y la batería dura bastante.", 1),
        ("La entrega fue antes de tiempo y todo venía sellado 👍", 1),
        ("Simplemente perfecto, vale cada centavo invertido :D", 1),
        ("Maravilloso rendimiento en tareas complejas.", 1),
        
        # Negativos
        ("El producto llegó roto y la caja toda aplastada :( 😡", 0),
        ("Pésimo servicio, tardó semanas y nunca respondieron mis correos.", 0),
        ("No es bueno, se calienta demasiado y se apaga solo 👎", 0),
        ("Una completa estafa, no sirve para nada y la calidad es basura 💩", 0),
        ("Muy mala experiencia, el personal fue déspota y grosero.", 0),
        ("No recomiendo este producto, material frágil y defectuoso 💔", 0),
        ("Nunca llegó mi paquete y se niegan a devolver el dinero 😡", 0),
        ("El sistema no funciona, tiene errores constantes y es lentísimo.", 0),
        ("No me gusta para nada, hace un ruido insoportable :( ", 0),
        ("Publicidad engañosa, el artículo no se parece en nada a las fotos.", 0),
        ("Pésima atención técnica, no resolvieron el fallo y me colgaron.", 0),
        ("La peor compra de mi vida, una pérdida total de dinero.", 0),
        ("No cumple con lo prometido, material de plástico corriente 👎", 0),
        ("Horrible servicio de paquetería y nula respuesta de soporte.", 0),
        ("Se descompuso al segundo día de uso, decepción absoluta 😢", 0)
    ]
    textos = [item[0] for item in data]
    etiquetas = [item[1] for item in data]
    return textos, etiquetas


# =====================================================================
# 5. ENTRENAMIENTO, EVALUACIÓN COMPARATIVA Y REPORTE
# =====================================================================

def imprimir_matriz_confusion_ascii(cm: np.ndarray, labels: List[str]):
    print("\n   Matriz de Confusión:")
    print(f"      Pred ->  {labels[0]:<10} {labels[1]:<10}")
    print(f"   Real {labels[0]:<6} [ {cm[0,0]:<8} {cm[0,1]:<8} ]")
    print(f"   Real {labels[1]:<6} [ {cm[1,0]:<8} {cm[1,1]:<8} ]\n")

def evaluar_clasificadores(X_train: np.ndarray, X_test: np.ndarray,
                           y_train: List[int], y_test: List[int],
                           nombre_extraccion: str):
    print("=" * 75)
    print(f"  EVALUACIÓN DE MODELOS CON EXTRACCIÓN: {nombre_extraccion}")
    print("=" * 75)
    
    modelos = {
        "Regresión Logística": LogisticRegression(random_state=42),
        "Support Vector Machine (SVM)": SVC(kernel='linear', C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    for nombre, clf in modelos.items():
        print(f"\n>>> Clasificador: {nombre}")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        reporte = classification_report(y_test, y_pred, target_names=["Negativo", "Positivo"], digits=3)
        print(reporte)
        
        cm = confusion_matrix(y_test, y_pred)
        imprimir_matriz_confusion_ascii(cm, ["Negativo", "Positivo"])


def main():
    print("=" * 75)
    print("  TECNM / ITSU - MAESTRÍA EN IA - ACTIVIDAD 16 (PLN SENTIMIENTOS)")
    print("=" * 75)
    
    # 1. Cargar datos
    textos, etiquetas = cargar_dataset_sentimientos()
    print(f"[*] Dataset cargado: {len(textos)} muestras balanceadas.")
    
    # 2. Demostración del preprocesamiento
    ejemplo = "El servicio no es bueno y el paquete nunca llegó :( 😡"
    ejemplo_limpio = limpiar_texto_espanol(ejemplo)
    print("\n[+] Demostración de Pipeline de Limpieza:")
    print(f"    Texto original: '{ejemplo}'")
    print(f"    Texto limpio:   '{ejemplo_limpio}'")
    
    # 3. Limpieza de todo el corpus
    textos_limpios = [limpiar_texto_espanol(t) for t in textos]
    
    # Split Train/Test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        textos_limpios, etiquetas, test_size=0.3, random_state=42, stratify=etiquetas
    )
    print(f"[*] Partición: {len(X_train_raw)} entrenamiento, {len(X_test_raw)} prueba.")
    
    # 4. Extracción TF-IDF
    print("\n--- 1. Extracción de Características: TF-IDF ---")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=100)
    X_train_tfidf = tfidf.fit_transform(X_train_raw).toarray()
    X_test_tfidf = tfidf.transform(X_test_raw).toarray()
    evaluar_clasificadores(X_train_tfidf, X_test_tfidf, y_train, y_test, "TF-IDF (scikit-learn)")
    
    # 5. Extracción Word2Vec Continuo
    print("\n--- 2. Extracción de Características: Word2Vec Continuo ---")
    w2v = ExtractorWord2Vec(vector_size=30, window=3, min_count=1)
    X_train_w2v = w2v.fit_transform(X_train_raw)
    X_test_w2v = w2v.transform(X_test_raw)
    evaluar_clasificadores(X_train_w2v, X_test_w2v, y_train, y_test, "Word2Vec Continuo (gensim)")
    
    print("\n" + "=" * 75)
    print("  CONSIDERACIONES ÉTICAS EN PROCESAMIENTO DE LENGUAJE NATURAL")
    print("=" * 75)
    print("""
    1. Sesgo Lingüístico y Dialectal: Los analizadores entrenados con español peninsular
       o neutro pueden fallar con modismos y dialectos regionales (ej. mexicano, andino).
    2. Privacidad de Datos: El análisis de opiniones y mensajes personales debe anonimizar
       datos personales identificables (PII) bajo regulaciones como RGPD y LFPDPPP.
    3. Moderación Ética: La clasificación errónea de sentimientos puede derivar en
       censura involuntaria o desatención de quejas legítimas de usuarios vulnerables.
    """)

if __name__ == "__main__":
    main()
