"""Tools for the fleet operations assistant.

Provides manual search and real-time UF value lookup.
"""

import requests
from src.rag_pipeline import build_retriever

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


def consultar_manual_operaciones(consulta: str) -> str:
    """Search the fleet operations manual for relevant information."""
    retriever = _get_retriever()
    docs = retriever(consulta, top_k=3)
    if not docs:
        return "No se encontraron resultados relevantes en el manual de operaciones."
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[Fragmento {i}]\n{doc}")
    return "\n\n".join(results)


def consultar_valor_uf_actual(reintentos: int = 3) -> str:
    """Fetch the current UF value from mindicador.cl API with retries."""
    url = "https://mindicador.cl/api/uf"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LogisticaExpressBot/1.0)"}
    ultimo_error = "sin respuesta"

    for intento in range(reintentos):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()

            # Current API format: {"serie": [{"fecha": ..., "valor": ...}, ...]}
            if "serie" in data and data["serie"]:
                uf_value = data["serie"][0]["valor"]
                uf_fecha = data["serie"][0]["fecha"]
            else:
                # Legacy format fallback: {"uf": {"valor": ..., "fecha": ...}}
                uf_value = data["uf"]["valor"]
                uf_fecha = data["uf"]["fecha"]

            fecha_corta = str(uf_fecha)[:10]
            formatted = f"${uf_value:,.2f} CLP"
            return f"Valor UF al {fecha_corta}: {formatted}"
        except requests.exceptions.Timeout:
            ultimo_error = "timeout"
        except requests.exceptions.RequestException as exc:
            ultimo_error = f"error de conexion: {exc}"
        except (KeyError, IndexError, ValueError) as exc:
            ultimo_error = f"formato inesperado: {exc}"

    return f"Error al consultar el valor de la UF tras {reintentos} intentos ({ultimo_error})."


TOOLS = {
    "consultar_manual_operaciones": consultar_manual_operaciones,
    "consultar_valor_uf_actual": consultar_valor_uf_actual,
}

TOOL_DESCRIPTIONS = """
Herramientas disponibles:
- consultar_manual_operaciones(consulta: str): Busca informacion en el manual de operaciones de Logistica Express. Usa para preguntas sobre procedimientos, fallas mecanicas, siniestros, jornada laboral, mantenimiento, combustible o talleres.
- consultar_valor_uf_actual(): Obtiene el valor actual de la UF (Unidad de Fomento) desde la API mindicador.cl. Usa para consultas economicas que involucren UF.
"""
