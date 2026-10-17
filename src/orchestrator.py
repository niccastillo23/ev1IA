"""Multi-agent orchestrator (supervisor) for Seguros Express.

Implements a supervisor + specialists pattern with LangGraph:

    START -> supervisor -> {coberturas | siniestros | reembolsos |
                            financiero | general} -> END

The supervisor classifies each query and delegates to a specialised ReAct
agent that owns a focused prompt and a tailored subset of tools. All agents
share the same conversational state (short-term memory via the graph
checkpointer) and the orchestrator can use a ``LongTermMemory`` for recall.
"""

import re
import unicodedata
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from src.agent_graph import build_model, sanitize_answer
from src.agent_tools import (
    buscar_manual,
    consultar_red_talleres,
    consultar_valor_uf_actual,
    convertir_uf_a_clp,
    registrar_siniestro,
)
from src.config import AGENT_RECURSION_LIMIT, CHAT_MODEL_SMALL, USE_LLM_ROUTER

CATEGORY_DESCRIPTIONS = {
    "coberturas": "Qué cubre o no la póliza: coberturas incluidas, exclusiones, robo, daños propios y a terceros.",
    "siniestros": "Protocolo ante accidentes/siniestros, denuncia en comisaría, asistencia en ruta, grúa y talleres.",
    "reembolsos": "Reembolsos de gastos y sus plazos/documentos, primas, pagos, mora, vigencia y renovación de la póliza.",
    "financiero": "Montos en UF o pesos: deducible, gastos médicos, conversión UF->CLP y valor vigente de la UF.",
    "general": "Saludos, consultas ambiguas o que no encajan en las categorías anteriores.",
}

SUPERVISOR_PROMPT = """\
Eres el supervisor de un equipo de agentes de Seguros Express. Clasifica la
consulta del asegurado en UNA categoría y delega en el especialista adecuado.

Categorías:
{categories}

Reglas:
- Si la consulta pide un monto en pesos o menciona valores UF (deducible,
  gastos médicos, conversión), elige "financiero".
- Si menciona coberturas o exclusiones (qué cubre / qué no cubre), elige "coberturas".
- Si menciona un accidente, choque, denuncia, grúa o taller, elige "siniestros".
- Si menciona reembolso, prima, pago, vigencia o renovación, elige "reembolsos".
- Ante saludos o consultas ambiguas, elige "general".
""".format(
    categories="\n".join(f'- "{name}": {desc}' for name, desc in CATEGORY_DESCRIPTIONS.items())
)


class RouteDecision(BaseModel):
    """Structured decision returned by the supervisor."""

    specialist: Literal["coberturas", "siniestros", "reembolsos", "financiero", "general"] = Field(
        description="Especialista que debe atender la consulta."
    )
    reason: str = Field(description="Breve justificación de la elección.")


class OrchestratorState(MessagesState):
    """Shared state: conversation plus the selected route and metadata."""

    route: str
    route_reason: str
    plan: list
    tools_used: list
    sources: list
    used_uf: bool
    uf_info: object


_SUPERVISOR = None


def _get_supervisor():
    global _SUPERVISOR
    if _SUPERVISOR is None:
        _SUPERVISOR = build_model(model=CHAT_MODEL_SMALL).with_structured_output(
            RouteDecision
        )
    return _SUPERVISOR


def _normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


_KEYWORDS = [
    (
        "financiero",
        ["uf", "unidad de fomento", "en pesos", "a pesos", "en clp", "en dinero", "equivalente"],
    ),
    (
        "coberturas",
        ["cubre", "cubren", "cobertura", "coberturas", "exclusion", "excluye", "excluido", "robo", "ampara", "amparo"],
    ),
    (
        "siniestros",
        ["siniestro", "accidente", "choque", "colision", "grua", "asistencia", "denuncia", "taller", "vuelco", "carabinero"],
    ),
    (
        "reembolsos",
        ["reembolso", "reembolsar", "prima", "primas", "pago", "pagar", "mora", "vigencia", "renovacion", "renovar", "cuota"],
    ),
]

_GREETINGS = ["hola", "buenos dias", "buenas tardes", "buenas noches", "buenas", "hey", "gracias"]

_FOLLOWUP_MARKERS = ["eso", "esa", "esto", "y ", "tambien", "ahora", "entonces", "aqui", "ahi"]

# Intent markers used by the planner to detect compound (multi-step) requests.
_WRITE_MARKERS = [
    "registrar", "registra", "reportar", "reporta", "declarar", "declara",
    "denunciar", "denuncia", "ingresar", "ingresa", "anotar", "anota",
]
_FINANCE_MARKERS = [
    "deducible", "en pesos", "a pesos", "en clp", "cuanto pagar", "cuanto pagare",
    "cuanto debo pagar", "monto", "valor uf", "en dinero", "cobertura en pesos",
]


def _build_plan(route: str, query: str) -> list:
    """Turn a query into an ordered list of specialists to run (plan).

    Simple queries map to a single specialist. Compound requests that mix a
    *write* intent (register a claim) with a *reasoning/economy* intent
    (how much the deductible costs) become an explicit two-step plan, so the
    agent first registers the siniestro and then computes the amount.
    """
    text = _normalize(query)
    write_hit = any(marker in text for marker in _WRITE_MARKERS)
    finance_hit = route == "financiero" or any(m in text for m in _FINANCE_MARKERS)

    if write_hit and finance_hit:
        return ["siniestros", "financiero"]
    if write_hit:
        return ["siniestros"]
    return [route]


def _is_greeting(text: str) -> bool:
    clean = _normalize(text)
    clean = "".join(c if (c.isalnum() or c.isspace()) else " " for c in clean)
    clean = " ".join(clean.split())
    return any(clean == _normalize(g) or clean.startswith(_normalize(g) + " ") for g in _GREETINGS)


def _keyword_route(query: str):
    """Deterministic first pass: route by domain keywords (priority order)."""
    text = _normalize(query)
    for specialist, keywords in _KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return specialist
    return None


def _raw_query(text: str) -> str:
    """Return the user's actual query, stripping any injected long-term memory."""
    text = str(text)
    if "[Consulta actual]" in text:
        return text.split("[Consulta actual]")[-1].strip()
    return text.strip()


def _classify(state: OrchestratorState) -> str:
    """Classify the latest user query.

    A deterministic keyword pass runs first (reliable for the common cases);
    the LLM supervisor only decides ambiguous queries. Short follow-ups
    ("¿y eso en pesos?") are combined with the previous user message so the
    thread context is preserved. Injected long-term memory is stripped so it
    cannot bias the routing.
    """
    user_messages = [
        m for m in state["messages"] if getattr(m, "type", "") == "human"
    ]
    if not user_messages:
        return "general"

    latest = _raw_query(user_messages[-1].content)
    if _is_greeting(latest):
        return "general"

    normalized = _normalize(latest)
    context = latest
    if (
        len(latest.split()) <= 6
        and len(user_messages) >= 2
        and any(marker in normalized for marker in _FOLLOWUP_MARKERS)
    ):
        context = f"{_raw_query(user_messages[-2].content)}\n{latest}"

    route = _keyword_route(context)
    if route:
        return route

    if not USE_LLM_ROUTER:
        return "general"

    try:
        decision = _get_supervisor().invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=context)]
        )
        return decision.specialist
    except Exception:  # noqa: BLE001 - never fail on routing
        return "general"


def _supervisor_node(state: OrchestratorState) -> dict:
    """Classify the query and build the execution plan (task sequencing)."""
    route = _classify(state)
    user_messages = [m for m in state["messages"] if getattr(m, "type", "") == "human"]
    query = _raw_query(user_messages[-1].content) if user_messages else ""
    plan = _build_plan(route, query)
    return {"route": route, "route_reason": "", "plan": plan}


_PATENTE_RE = re.compile(r"\b([A-Za-z]{2,4}[\s-]?\d{2,4})\b")
_CIUDAD_RE = re.compile(r"\ben\s+([A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)")
_UF_AMOUNT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*UF", re.IGNORECASE)
_SINIESTRO_TIPOS = [
    "choque", "colision", "robo", "vuelco", "incendio", "accidente",
    "granizo", "inundacion", "hurto",
]

COMPOSE_SYSTEM = """\
Eres el Asistente Inteligente de Seguros Express S.A. Redactas la respuesta
final al asegurado en español, de forma clara, breve y profesional.

Reglas estrictas:
- Responde únicamente con la información entregada (fragmentos del manual y
  acciones/valores ya obtenidos).
- No menciones herramientas, funciones, procesos internos, ni que "vas a
  buscar" o "necesitas consultar": la información ya te fue entregada.
- No inventes cifras, teléfonos, nombres ni montos. Si un dato no aparece, di
  que no cuentas con esa información.
- Copia literalmente los términos del manual (por ejemplo "Unidades de
  Fomento", "800-500-100", "5 UF").
- Si la consulta trata sobre coberturas o exclusiones de la póliza (por ejemplo
  conducción en estado de ebriedad o bajo efectos de drogas), responde de forma
  neutral y factual indicando si está cubierto o excluido según el manual: es
  una consulta sobre la póliza, no una recomendación de conducta.
- Si entre las acciones figura un registro o un folio, menciónalo brevemente.
- Si las acciones/valores ya contienen el dato solicitado, entrégalo
  directamente, sin decir que no lo tienes.
- Habla como el asistente en primera persona (por ejemplo "No cuento con esa
  información"), dirigiendo la respuesta al asegurado."""

# The local model can refuse policy questions that mention sensitive topics
# (e.g. driving under the influence). When that happens we fall back to an
# extractive answer taken verbatim from the manual, which is deterministic.
_REFUSAL_MARKERS = [
    "no puedo proporcionar",
    "no puedo asistir",
    "no puedo ayudar",
    "no puedo brindar",
    "no es apropiado",
    "actividades ilegales",
    "fines ilegales",
    "contenido que pueda ser utilizado",
]


def _is_refusal(text: str) -> bool:
    lowered = _normalize(text or "")
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def _extractive_fallback(query: str, context: str) -> str:
    """Deterministic answer built from the most relevant manual sentence."""
    terms = [w for w in _normalize(query).split() if len(w) >= 4]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", context or "") if s.strip()]
    if not sentences:
        return "No cuento con esa información en el manual del asegurado."

    def score(sentence: str) -> int:
        normalized = _normalize(sentence)
        return sum(term in normalized for term in terms)

    best = max(sentences, key=score)
    best = " ".join(best.split())
    text = f"Según el manual del asegurado: {best}"
    index = (context or "").find(best)
    window = _normalize((context or "")[max(0, index - 200):index])
    if "exclu" in window and "no cubre" not in window:
        text += " Por lo tanto, esta situación figura entre las exclusiones y no está cubierta por la póliza."
    return text

COMPOSE_TEMPLATE = """\
[Fragmentos del manual]
{context}

[Acciones y valores obtenidos]
{facts}

[Consulta del asegurado]
{query}

Respuesta:"""


def _latest_query(state: OrchestratorState) -> str:
    user_messages = [m for m in state["messages"] if getattr(m, "type", "") == "human"]
    return _raw_query(user_messages[-1].content) if user_messages else ""


def _contextual_query(state: OrchestratorState) -> str:
    """Latest query, prefixed with the previous turn when it is a short follow-up.

    Enables context carry-over (e.g. "¿Cuánto es el deducible?" ... "¿y eso en
    pesos?") so the deterministic tools can resolve the referent.
    """
    user_messages = [m for m in state["messages"] if getattr(m, "type", "") == "human"]
    if not user_messages:
        return ""
    latest = _raw_query(user_messages[-1].content)
    if len(user_messages) >= 2:
        normalized = _normalize(latest)
        if len(latest.split()) <= 6 and any(m in normalized for m in _FOLLOWUP_MARKERS):
            previous = _raw_query(user_messages[-2].content)
            return f"{previous} {latest}"
    return latest


def _extract_patente(query: str):
    match = _PATENTE_RE.search(query)
    return match.group(1) if match else None


def _extract_tipo(normalized: str):
    for tipo in _SINIESTRO_TIPOS:
        if tipo in normalized:
            return tipo
    return None


def _extract_ciudad(query: str):
    match = _CIUDAD_RE.search(query)
    return match.group(1) if match else None


def _find_uf_amount(query: str, context: str):
    """Find the relevant UF amount: deducible first, else first UF in context."""
    for source in (query, context):
        match = re.search(
            r"deducible[^.]{0,40}?(\d+(?:[.,]\d+)?)\s*UF", source or "", re.IGNORECASE
        )
        if match:
            return float(match.group(1).replace(",", "."))
        match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*UF[^.]{0,40}?deducible", source or "", re.IGNORECASE
        )
        if match:
            return float(match.group(1).replace(",", "."))
    match = _UF_AMOUNT_RE.search(context or "") or _UF_AMOUNT_RE.search(query or "")
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def _run_tools(plan: list, query: str) -> dict:
    """Execute the plan's tools deterministically (grounding + write).

    This guarantees the tools actually run (query + write + reasoning) regardless
    of how reliable the LLM's own tool-calling is, then the model only composes
    the answer from the returned data.
    """
    text = _normalize(query)
    write_hit = any(marker in text for marker in _WRITE_MARKERS)

    context = ""
    sources = []
    facts = []
    tools_used = []
    used_uf = False
    uf_info = None

    manual = buscar_manual.invoke({"consulta": query})
    tools_used.append("buscar_manual")
    if manual and "No se encontró" not in manual:
        context = manual
        sources.append(manual)

    if "taller" in text:
        taller = consultar_red_talleres.invoke({"ciudad": _extract_ciudad(query) or query})
        facts.append(taller)
        tools_used.append("consultar_red_talleres")

    if "siniestros" in plan and write_hit:
        patente = _extract_patente(query)
        tipo = _extract_tipo(text)
        if patente and tipo:
            result = registrar_siniestro.invoke(
                {
                    "patente": patente,
                    "tipo": tipo,
                    "descripcion": query,
                    "ciudad": _extract_ciudad(query) or "",
                }
            )
            facts.append(result)
            tools_used.append("registrar_siniestro")
        else:
            facts.append(
                "No se registró el siniestro: faltan la patente y/o el tipo "
                "(choque, robo, vuelco, etc.)."
            )

    if "financiero" in plan:
        uf_value = consultar_valor_uf_actual.invoke({})
        tools_used.append("consultar_valor_uf_actual")
        used_uf = True
        uf_info = uf_value
        amount = _find_uf_amount(query, context)
        if amount is not None:
            conversion = convertir_uf_a_clp.invoke({"cantidad_uf": amount})
            facts.append(conversion)
            tools_used.append("convertir_uf_a_clp")

    return {
        "context": context,
        "sources": sources,
        "facts": facts,
        "tools_used": tools_used,
        "used_uf": used_uf,
        "uf_info": uf_info,
    }


def _compose_answer(query: str, context: str, facts: list) -> str:
    """Single grounded LLM call that writes the final answer (no tool loop)."""
    prompt = (
        f"{COMPOSE_SYSTEM}\n\n"
        + COMPOSE_TEMPLATE.format(
            context=context or "(no se encontraron fragmentos en el manual)",
            facts="\n".join(f"- {f}" for f in facts) or "(sin acciones ni valores)",
            query=query,
        )
    )
    response = build_model().invoke(prompt)
    answer = sanitize_answer(str(getattr(response, "content", "")))
    if _is_refusal(answer) and context:
        return _extractive_fallback(query, context)
    return answer


def _executor_node(state: OrchestratorState) -> dict:
    """Run the plan: execute tools deterministically, then compose the answer.

    The supervisor produced an ordered plan (task sequencing). For each plan the
    executor runs the required tools (manual RAG, siniestro registration, UF
    lookup and UF->CLP conversion) and merges their results, then a single LLM
    call writes a grounded answer. This keeps multi-stage behaviour reliable and
    fast while preserving the supervisor + specialists design.
    """
    plan = state.get("plan") or [state.get("route") or "general"]
    query = _latest_query(state)
    tool_query = _contextual_query(state)

    outcome = _run_tools(plan, tool_query)
    answer = _compose_answer(query, outcome["context"], outcome["facts"])

    return {
        "messages": [AIMessage(content=answer)],
        "tools_used": outcome["tools_used"],
        "sources": outcome["sources"],
        "used_uf": outcome["used_uf"],
        "uf_info": outcome["uf_info"],
    }


def build_orchestrator(checkpointer=None):
    """Build and compile the supervisor + planner + executor graph.

    ``supervisor`` classifies the query and produces a plan; ``executor`` runs
    each planned specialist in order. Single-intent queries yield a one-step
    plan (equivalent to the previous behaviour).
    """
    builder = StateGraph(OrchestratorState)
    builder.add_node("supervisor", _supervisor_node)
    builder.add_node("executor", _executor_node)

    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "executor")
    builder.add_edge("executor", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def _augment_with_long_term(query: str, long_term) -> str:
    if long_term is None:
        return query
    recalled = long_term.recall(query)
    if not recalled:
        return query
    memoria = "\n".join(f"- {item}" for item in recalled)
    return (
        "[Contexto de conversaciones previas del asegurado]\n"
        f"{memoria}\n\n[Consulta actual]\n{query}"
    )


def run_orchestrator(graph, query: str, thread_id: str = "default", long_term=None) -> dict:
    """Run one turn through the supervisor graph and return answer + metadata."""
    augmented = _augment_with_long_term(query, long_term)
    initial_state = {
        "messages": [("user", augmented)],
        "route": "",
        "route_reason": "",
        "plan": [],
        "tools_used": [],
        "sources": [],
        "used_uf": False,
        "uf_info": None,
    }
    result = graph.invoke(
        initial_state,
        config={
            "configurable": {"thread_id": thread_id},
            "recursion_limit": AGENT_RECURSION_LIMIT,
        },
    )
    messages = result["messages"]
    answer = messages[-1].content
    if not isinstance(answer, str):
        answer = str(answer)
    answer = sanitize_answer(answer)

    if long_term is not None:
        long_term.remember(f"Pregunta: {query}\nRespuesta: {answer}")

    return {
        "answer": answer,
        "route": result.get("route", ""),
        "route_reason": result.get("route_reason", ""),
        "plan": result.get("plan", []),
        "tools_used": result.get("tools_used", []),
        "sources": result.get("sources", []),
        "used_uf": result.get("used_uf", False),
        "uf_info": result.get("uf_info"),
    }
