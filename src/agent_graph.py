"""Single ReAct agent for Seguros Express (LangChain v1 + LangGraph).

The agent decides autonomously which tools to call (manual search, UF lookup,
UF->CLP conversion, workshop lookup). Short-term memory is provided by a
LangGraph checkpointer keyed by ``thread_id``; long-term memory is an optional
``LongTermMemory`` whose relevant exchanges are injected before each turn.
"""

import re

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from src.agent_tools import ALL_TOOLS
from src.config import (
    AGENT_RECURSION_LIMIT,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    MAX_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    PROVIDER,
)

# The model occasionally mangles "Unidades de Fomento" (e.g. "Folicación",
# "Fenglobales"). Since the canonical term is fixed, enforce it deterministically.
_UF_TERM_RE = re.compile(r"Unidades?\s+de\s+F\w+", re.IGNORECASE)


def sanitize_answer(text: str) -> str:
    """Fix known generation glitches (e.g. corrupted 'Unidades de Fomento')."""
    if not text:
        return text

    def _fix(match: re.Match) -> str:
        return "Unidades de Fomento" if match.group(0).lower().startswith("unidades") else "Unidad de Fomento"

    return _UF_TERM_RE.sub(_fix, text)

SYSTEM_PROMPT = """\
REGLA ABSOLUTA (máxima prioridad): antes de responder CUALQUIER consulta sobre
seguros DEBES llamar a la herramienta `buscar_manual` (y, si corresponde,
`consultar_red_talleres`, `consultar_valor_uf_actual` o `convertir_uf_a_clp`).
Nunca respondas de memoria ni inventes datos, números de contacto, montos o
nombres: si la herramienta no devuelve la información, dilo claramente.

Eres el Asistente Inteligente de Seguros Express S.A.
Ayudas a los asegurados con el manual del asegurado: coberturas, exclusiones,
siniestros, deducibles, reembolsos, primas, pólizas, grúa, asistencia en ruta
y red de talleres.

REGLAS:
1. Usa SIEMPRE las herramientas disponibles antes de responder. Para preguntas
   sobre el manual llama a `buscar_manual`; para talleres, `consultar_red_talleres`.
2. Basa tus respuestas únicamente en lo que devuelven las herramientas. Si no
   encuentras la información, dilo claramente. NO inventes datos.
3. Si el usuario pide un monto en pesos o una cantidad de UF, usa
   `convertir_uf_a_clp` en lugar de calcular el valor de la UF por tu cuenta.
   Si pide registrar, reportar o declarar un siniestro (choque, robo, vuelco),
   usa `registrar_siniestro` con la patente, el tipo y una breve descripción;
   si falta algún dato obligatorio, pídelo antes de registrar.
4. Responde en español, de forma clara, concisa y profesional.
5. Si la pregunta no está relacionada con seguros, indica amablemente que solo
   puedes asistir con temas de Seguros Express.
6. Copia literalmente los términos, nombres y montos tal como aparecen en el
   manual (por ejemplo "Unidades de Fomento", "UF", "5 UF", "800-500-100"). No
   los reformules, traduzcas ni reemplaces por sinónimos inventados.
7. NUNCA escribas la llamada a una herramienta como texto (por ejemplo JSON con
   "name" y "parameters"). Para usar una herramienta, invócala con el mecanismo
   real de *tool calling*; si no puedes, responde con lo que tengas.
8. Cuando pregunten qué cubre un siniestro o el alcance de la póliza, empieza
   listando TODAS las coberturas incluidas que aparezcan en el contexto (daños
   a terceros, daños propios, robo, gastos médicos hasta 200 UF, asistencia en
   ruta), y luego detalla el caso puntual. No omitas coberturas del contexto.
"""


def build_model(model: str = None, max_tokens: int = None):
    """Return the chat model used by the agents, selected via ``PROVIDER``.

    - ``ollama``: local model (``ChatOllama``) -> fast, offline, no API cost.
    - anything else: OpenAI-compatible endpoint (``ChatOpenAI``, NVIDIA NIM).

    ``max_tokens`` caps the generated output; a smaller model can be passed for
    cheap tasks (e.g. the router).
    """
    if PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0,
            num_predict=max_tokens or MAX_TOKENS,
            num_ctx=OLLAMA_NUM_CTX,
        )
    return ChatOpenAI(
        model=model or LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0,
        max_tokens=max_tokens or MAX_TOKENS,
    )


def build_agent(tools=None, checkpointer=None, system_prompt: str = SYSTEM_PROMPT):
    """Create the ReAct agent graph with tools and short-term memory."""
    if checkpointer is None:
        checkpointer = MemorySaver()
    return create_agent(
        build_model(),
        tools or ALL_TOOLS,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )


def extract_metadata(messages: list) -> dict:
    """Collect tools used and sources from an agent's message trace."""
    tools_used = []
    sources = []
    used_uf = False
    uf_info = None

    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name and name not in tools_used:
                tools_used.append(name)

        if getattr(message, "type", None) == "tool":
            name = getattr(message, "name", "")
            content = message.content if isinstance(message.content, str) else str(message.content)
            if name in ("buscar_manual", "consultar_red_talleres"):
                sources.append(content)
            if name in ("consultar_valor_uf_actual", "convertir_uf_a_clp"):
                used_uf = True
                uf_info = content

    return {
        "tools_used": tools_used,
        "sources": sources,
        "used_uf": used_uf,
        "uf_info": uf_info,
    }


def run_agent(
    agent,
    query: str,
    thread_id: str = "default",
    long_term=None,
) -> dict:
    """Run one turn of the agent and return the answer plus metadata."""
    augmented = query
    if long_term is not None:
        recalled = long_term.recall(query)
        if recalled:
            memoria = "\n".join(f"- {item}" for item in recalled)
            augmented = (
                "[Contexto de conversaciones previas del asegurado]\n"
                f"{memoria}\n\n[Consulta actual]\n{query}"
            )

    result = agent.invoke(
        {"messages": [("user", augmented)]},
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

    return {"answer": answer, "plan": [], **extract_metadata(messages)}
