"""Multi-agent orchestrator (supervisor) for Seguros Express.

Implements a supervisor + specialists pattern with LangGraph:

    START -> supervisor -> {coberturas | siniestros | reembolsos |
                            financiero | general} -> END

The supervisor classifies each query and delegates to a specialised ReAct
agent that owns a focused prompt and a tailored subset of tools. All agents
share the same conversational state (short-term memory via the graph
checkpointer) and the orchestrator can use a ``LongTermMemory`` for recall.
"""

import unicodedata
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from src.agent_graph import SYSTEM_PROMPT, build_model, extract_metadata, sanitize_answer
from src.agent_tools import (
    ALL_TOOLS,
    buscar_manual,
    consultar_red_talleres,
    consultar_valor_uf_actual,
    convertir_uf_a_clp,
)

SPECIALIST_NAMES = ["coberturas", "siniestros", "reembolsos", "financiero", "general"]

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
    tools_used: list
    sources: list
    used_uf: bool
    uf_info: object


TOOLS_BY_SPECIALIST = {
    "coberturas": [buscar_manual],
    "siniestros": [buscar_manual, consultar_red_talleres],
    "reembolsos": [buscar_manual],
    "financiero": [buscar_manual, consultar_valor_uf_actual, convertir_uf_a_clp],
    "general": ALL_TOOLS,
}

ROLE_BY_SPECIALIST = {
    "coberturas": (
        "Te especializas en COBERTURAS Y EXCLUSIONES. Explica con precisión qué "
        "cubre la póliza y qué queda excluido, citando el manual. Si el contexto "
        "incluye la lista de coberturas incluidas (daños a terceros, daños "
        "propios, robo, gastos médicos hasta 200 UF, asistencia en ruta), "
        "reprodúcela COMPLETA antes de detallar el caso puntual; no te limites a "
        "la cobertura más directa."
    ),
    "siniestros": (
        "Te especializas en SINIESTROS Y ASISTENCIA EN RUTA. Guía al asegurado "
        "paso a paso (protocolo, denuncia, grúa) y entrega datos de la red de talleres."
    ),
    "reembolsos": (
        "Te especializas en REEMBOLSOS, PRIMAS Y VIGENCIA. Indica plazos, "
        "documentos requeridos y condiciones de pago o renovación."
    ),
    "financiero": (
        "Te especializas en MONTOS ECONÓMICOS. Convierte valores de UF a pesos "
        "usando las herramientas y explica deducibles y coberturas en dinero."
    ),
    "general": (
        "Atiendes consultas generales y saludos. Si la pregunta es sobre el "
        "manual, búscala; si no está en el contexto, dilo claramente."
    ),
}

_SPECIALIST_AGENTS = {}
_SUPERVISOR = None


def _get_supervisor():
    global _SUPERVISOR
    if _SUPERVISOR is None:
        _SUPERVISOR = build_model().with_structured_output(RouteDecision)
    return _SUPERVISOR


def _get_specialist_agent(name: str):
    if name not in _SPECIALIST_AGENTS:
        system_prompt = f"{SYSTEM_PROMPT}\n\nROL DEL ESPECIALISTA:\n{ROLE_BY_SPECIALIST[name]}"
        _SPECIALIST_AGENTS[name] = create_agent(
            build_model(),
            TOOLS_BY_SPECIALIST[name],
            system_prompt=system_prompt,
        )
    return _SPECIALIST_AGENTS[name]


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

    try:
        decision = _get_supervisor().invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=context)]
        )
        return decision.specialist
    except Exception:  # noqa: BLE001 - never fail on routing
        return "general"


def _supervisor_node(state: OrchestratorState) -> dict:
    route = _classify(state)
    return {"route": route, "route_reason": ""}


def _make_specialist_node(name: str):
    def node(state: OrchestratorState) -> dict:
        agent = _get_specialist_agent(name)
        result = agent.invoke({"messages": state["messages"]})
        messages = result["messages"]
        metadata = extract_metadata(messages)
        return {"messages": [messages[-1]], **metadata}

    return node


def build_orchestrator(checkpointer=None):
    """Build and compile the supervisor graph."""
    builder = StateGraph(OrchestratorState)
    builder.add_node("supervisor", _supervisor_node)
    for name in SPECIALIST_NAMES:
        builder.add_node(name, _make_specialist_node(name))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["route"],
        {name: name for name in SPECIALIST_NAMES},
    )
    for name in SPECIALIST_NAMES:
        builder.add_edge(name, END)

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
        "tools_used": [],
        "sources": [],
        "used_uf": False,
        "uf_info": None,
    }
    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
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
        "tools_used": result.get("tools_used", []),
        "sources": result.get("sources", []),
        "used_uf": result.get("used_uf", False),
        "uf_info": result.get("uf_info"),
    }
