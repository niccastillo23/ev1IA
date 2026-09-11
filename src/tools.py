"""LangChain tools for the fleet operations assistant.

Provides semantic search over the operations manual and
real-time UF (Unidad de Fomento) value lookup.
"""

import requests
from langchain_core.tools import tool

from src.rag_pipeline import build_retriever

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


@tool
def consultar_manual_operaciones(consulta: str) -> str:
    """Search the fleet operations manual for information about
    mechanical failures, accidents, driving hours, maintenance,
    fuel policy, or authorized workshops.

    Args:
        consulta: The user's question or search query in Spanish.

    Returns:
        Relevant excerpts from the operations manual.
    """
    retriever = _get_retriever()
    docs = retriever.invoke(consulta)
    if not docs:
        return "No se encontraron resultados relevantes en el manual de operaciones."
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"[Fragmento {i}]\n{doc.page_content}")
    return "\n\n".join(results)


@tool
def consultar_valor_uf_actual() -> str:
    """Fetch the current UF (Unidad de Fomento) value from the
    mindicador.cl API and return it formatted in CLP.

    Returns:
        The current UF value with its date, formatted in CLP.
    """
    url = "https://mindicador.cl/api/uf"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        uf_value = data["uf"]["valor"]
        uf_fecha = data["uf"]["fecha"]
        formatted = f"${uf_value:,.2f} CLP"
        return f"Valor UF al {uf_fecha}: {formatted}"
    except requests.exceptions.Timeout:
        return "Error: timeout al consultar el valor de la UF. Intente nuevamente."
    except requests.exceptions.RequestException as exc:
        return f"Error al consultar el valor de la UF: {exc}"
    except (KeyError, ValueError) as exc:
        return f"Error al procesar la respuesta de la API: {exc}"
