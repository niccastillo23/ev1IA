"""LangChain tools for the Seguros Express agent.

Thin wrappers around the manual retriever and the external UF API, plus a
couple of derived tools (UF -> CLP conversion and workshop lookup) that the
agent can call autonomously. It also exposes a *write* tool that persists a
siniestro to disk, so the agent covers query + reasoning + writing.
"""

import json
import re
from datetime import datetime

from langchain_core.tools import tool

from src.config import SINIESTROS_DB_PATH
from src.rag_pipeline import build_retriever
from src.tools import consultar_valor_uf_actual as _consultar_uf

_retriever = None


def _get_retriever():
    """Lazily build the (hybrid) retriever once per process."""
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


def _format_docs(docs: list) -> str:
    return "\n\n".join(f"[Fragmento {i}]\n{doc}" for i, doc in enumerate(docs, 1))


def _uf_number(raw: str):
    """Extract the numeric UF value from the tool's formatted string."""
    cleaned = raw.replace(",", "")
    match = re.search(r"\$\s*([\d.]+)", cleaned)
    return float(match.group(1)) if match else None


@tool
def buscar_manual(consulta: str) -> str:
    """Busca información en el manual del asegurado de Seguros Express.

    Úsala para preguntas sobre coberturas, exclusiones, siniestros,
    deducibles, reembolsos, primas, pólizas, grúa, asistencia en ruta o red
    de talleres. Devuelve los fragmentos más relevantes del manual.
    """
    docs = _get_retriever()(consulta, top_k=4)
    if not docs:
        return "No se encontró información relevante en el manual del asegurado."
    return _format_docs(docs)


@tool
def consultar_valor_uf_actual() -> str:
    """Obtiene el valor vigente de la UF (Unidad de Fomento) desde mindicador.cl.

    Úsala cuando el asegurado pregunte por el valor de la UF o por equivalencias
    a pesos chilenos que requieran el valor actual.
    """
    return _consultar_uf()


@tool
def convertir_uf_a_clp(cantidad_uf: float) -> str:
    """Convierte una cantidad de UF a pesos chilenos (CLP) con el valor vigente.

    Úsala para resolver montos como el deducible (5 UF) o coberturas médicas
    (hasta 200 UF) en pesos. Ejemplo: convertir_uf_a_clp(5).
    """
    raw = _consultar_uf()
    value = _uf_number(raw)
    if value is None:
        return f"No se pudo obtener el valor de la UF. Detalle: {raw}"
    monto = cantidad_uf * value
    return f"{cantidad_uf} UF = ${monto:,.2f} CLP ({raw})."


@tool
def consultar_red_talleres(ciudad: str) -> str:
    """Busca talleres y prestadores autorizados de la red por ciudad.

    Úsala para consultas sobre direcciones, horarios o teléfonos de talleres
    (por ejemplo Santiago, Valparaíso o Concepción).
    """
    docs = _get_retriever()(f"taller prestador autorizado {ciudad}", top_k=2)
    if not docs:
        return f"No se encontraron talleres para la ciudad '{ciudad}' en el manual."
    return _format_docs(docs)


def _load_siniestros() -> list:
    """Read the siniestros store (returns [] if missing/corrupt)."""
    path = SINIESTROS_DB_PATH
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_siniestros(items: list) -> None:
    path = SINIESTROS_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_folio(items: list) -> str:
    """Sequential folio per day, e.g. SIN-20260116-0003."""
    day = datetime.now().strftime("%Y%m%d")
    prefix = f"SIN-{day}-"
    seq = sum(1 for item in items if str(item.get("folio", "")).startswith(prefix)) + 1
    return f"{prefix}{seq:04d}"


@tool
def registrar_siniestro(patente: str, tipo: str, descripcion: str, ciudad: str = "") -> str:
    """Registra (ESCRITURA) un siniestro y devuelve un folio de seguimiento.

    Úsala cuando el asegurado pida reportar, declarar, denunciar o registrar un
    siniestro (choque, robo, vuelco, etc.). Persiste el registro en el sistema.
    """
    items = _load_siniestros()
    folio = _next_folio(items)
    items.append(
        {
            "folio": folio,
            "patente": patente.strip().upper(),
            "tipo": tipo.strip(),
            "descripcion": descripcion.strip(),
            "ciudad": (ciudad or "no indicada").strip(),
            "estado": "registrado",
            "fecha": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save_siniestros(items)
    return (
        f"Siniestro registrado con folio {folio}. "
        f"Patente {patente.strip().upper()}, tipo {tipo.strip()}, estado: registrado."
    )


@tool
def consultar_siniestros(patente: str) -> str:
    """Consulta los siniestros ya registrados para una patente.

    Úsala para verificar el estado o el historial de siniestros de un asegurado.
    """
    target = patente.strip().upper()
    items = [it for it in _load_siniestros() if str(it.get("patente", "")).upper() == target]
    if not items:
        return f"No hay siniestros registrados para la patente {target}."
    lines = [
        f"- {it.get('folio')}: {it.get('tipo')} ({it.get('estado')}) el {it.get('fecha')}"
        for it in items
    ]
    return f"Siniestros registrados para {target}:\n" + "\n".join(lines)


ALL_TOOLS = [
    buscar_manual,
    consultar_valor_uf_actual,
    convertir_uf_a_clp,
    consultar_red_talleres,
    registrar_siniestro,
    consultar_siniestros,
]
