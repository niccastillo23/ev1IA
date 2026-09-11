"""RAG pipeline for the fleet operations assistant.

Implements lexical hybrid retrieval over the operations manual:
text normalization (accents), Spanish stopwords, light stemming
(prefix matching) and a domain synonym map. Uses only the standard
library, so no embedding provider is required.
"""

import re
import unicodedata

from src.config import MANUAL_PATH

STOPWORDS = {
    "de", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "y", "o", "u", "para", "por", "con", "sin", "que", "cual",
    "cuales", "como", "cuanto", "cuanta", "cuantos", "cuantas",
    "es", "son", "esta", "estan", "al", "del", "en", "a", "se",
    "su", "sus", "mi", "mis", "tu", "tus", "me", "te", "lo",
    "le", "nos", "hay", "si", "no", "mas", "pero", "sobre",
    "tiene", "tengo", "puedo", "debo", "hago", "hacer", "donde",
    "cuando", "porque", "ser", "sera", "fue", "era",
}

SYNONYMS = {
    "conducir": {"conduccion", "manejo", "manejar", "conduc"},
    "conduccion": {"conducir", "manejo", "conduc"},
    "manejo": {"conduccion", "conducir"},
    "manejar": {"conduccion", "conducir"},
    "limite": {"maximo", "maxima", "tope"},
    "limites": {"maximo", "maxima", "tope"},
    "maximo": {"limite", "tope"},
    "grua": {"auxilio", "remolque", "asistencia"},
    "auxilio": {"grua", "remolque"},
    "telefono": {"numero", "contacto", "central"},
    "numero": {"telefono", "contacto"},
    "siniestro": {"accidente", "choque", "colision"},
    "accidente": {"siniestro", "choque"},
    "combustible": {"bencina", "diesel", "gasolina", "carga"},
    "descanso": {"pausa", "receso", "jornada"},
    "jornada": {"descanso", "pausa", "horario"},
    "mantenimiento": {"revision", "service", "preventivo"},
    "taller": {"talleres", "servicio", "reparacion"},
    "deducible": {"seguro", "cobertura"},
    "seguro": {"deducible", "cobertura"},
}


def normalize(text: str) -> str:
    """Lowercase and remove accents (á -> a, ñ -> n, etc.)."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def _stem(token: str) -> str:
    """Light stemmer: keep the first 5 characters of the token."""
    return token[:5]


def _tokenize(text: str) -> set:
    """Normalize, extract word tokens and drop stopwords/short words."""
    tokens = re.findall(r"\w+", normalize(text))
    return {t for t in tokens if t not in STOPWORDS and len(t) > 2}


def _expand(tokens: set) -> set:
    """Add domain synonyms to the query token set."""
    expanded = set(tokens)
    for token in tokens:
        expanded.update(SYNONYMS.get(token, set()))
    return expanded


def _matches(a: str, b: str) -> bool:
    """Prefix-based match so 'grua' ~ 'gruas' and 'condu' ~ 'conduc'."""
    if a == b:
        return True
    return len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a))


def _score(query_stems: set, doc: str, title: str) -> int:
    """Score a document by matching query stems, boosting title matches."""
    doc_stems = {_stem(t) for t in _tokenize(doc)}
    title_stems = {_stem(t) for t in _tokenize(title)}

    score = 0
    for query_stem in query_stems:
        if any(_matches(query_stem, d) for d in doc_stems):
            score += 1
            if any(_matches(query_stem, t) for t in title_stems):
                score += 1
    return score


def load_and_split_manual() -> list:
    """Load the operations manual and split it into section chunks."""
    with open(MANUAL_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r"\n={2,}\n", content)
    return [section.strip() for section in sections if section.strip()]


def keyword_retrieval(query: str, documents: list, top_k: int = 3) -> list:
    """Retrieve the top_k most relevant chunks for a query.

    Returns an empty list when no content word matches, preserving the
    anti-hallucination behavior.
    """
    query_stems = {_stem(t) for t in _expand(_tokenize(query))}
    if not query_stems:
        return []

    scored = []
    for doc in documents:
        title = doc.splitlines()[0] if doc else ""
        score = _score(query_stems, doc, title)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def build_retriever():
    """Load the manual and return a retriever function."""
    documents = load_and_split_manual()

    def retriever(query: str, top_k: int = 3) -> list:
        return keyword_retrieval(query, documents, top_k)

    return retriever
